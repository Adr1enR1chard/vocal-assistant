"""Ear and voice: microphone, wake word, recording until silence, STT and TTS."""

from __future__ import annotations

import queue
import re
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import Settings
from .metrics import Turn

RATE = 16_000
FRAME = 1280  # 80 ms: the block size openWakeWord expects


def _device(value: str | None):
    return int(value) if value and value.isdigit() else value


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


class Microphone:
    """Continuous microphone stream (16 kHz mono int16) split into 80 ms blocks.

    Also tracks the ambient noise level over the last few seconds of frames read, so that a
    recording that starts while the user is already talking still knows what silence sounds like.
    """

    def __init__(self, settings: Settings):
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self.recent_rms: deque[float] = deque(maxlen=50)  # 4 s
        self.stream = sd.InputStream(
            samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME,
            device=_device(settings.input_device),
            callback=lambda data, *_: self.frames.put(data[:, 0].copy()),
        )

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def flush(self) -> None:
        while not self.frames.empty():
            self.frames.get_nowait()

    def read(self) -> np.ndarray:
        frame = self.frames.get()
        self.recent_rms.append(_rms(frame))
        return frame

    def noise_floor(self) -> float | None:
        """A low percentile of recent levels: the quiet moments, not the speech."""
        if len(self.recent_rms) < 5:
            return None
        return float(np.percentile(self.recent_rms, 20))


class WakeWord:
    def __init__(self, settings: Settings):
        from openwakeword.model import Model

        if settings.wake_word_is_custom and not Path(settings.wake_word).exists():
            raise FileNotFoundError(
                f"Wake word model not found: {settings.wake_word}. Train it first (see README, "
                "\"Wake word\"), use `--ptt`, or set WAKE_WORD=hey_jarvis meanwhile."
            )
        self.threshold = settings.wake_threshold
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, round(settings.wake_preroll_s * RATE / FRAME)))
        self.keep_preroll = settings.wake_preroll_s > 0
        self.model = Model(wakeword_models=[settings.wake_word], inference_framework="onnx")

    def wait(self, mic: Microphone) -> tuple[float, list[np.ndarray]]:
        """Blocks until the wake word. Returns its score and the last frames heard.

        The detection fires a little after the word ends: when the question follows without a
        pause, its first syllables are in those frames, so the recording starts with them.
        """
        mic.flush()
        self.model.reset()
        self.preroll.clear()
        while True:
            frame = mic.read()
            self.preroll.append(frame)
            score = max(self.model.predict(frame).values())
            if score >= self.threshold:
                return score, list(self.preroll) if self.keep_preroll else []


def record_utterance(mic: Microphone, settings: Settings, no_speech_timeout_s: float | None = None,
                     preroll: list[np.ndarray] | None = None) -> np.ndarray | None:
    """Records until the user stops talking. Energy threshold adapts to the ambient noise.

    The noise floor comes from what the mic heard before (not from the first frame, which is
    already speech when the question follows the wake word). Speech counts only after
    `min_speech_s` of voiced frames, so a click or the tail of the wake word doesn't end the
    recording before the question. `preroll` frames are kept in the audio but not analysed.
    """
    timeout = settings.no_speech_timeout_s if no_speech_timeout_s is None else no_speech_timeout_s
    frames: list[np.ndarray] = list(preroll or [])
    rms_floor = mic.noise_floor()
    voiced = 0.0
    speaking = False
    silence = 0.0
    started = time.monotonic()
    frame_s = FRAME / RATE
    while True:
        frame = mic.read()
        frames.append(frame)
        rms = _rms(frame)
        if rms_floor is None:
            rms_floor = rms
        threshold = max(rms_floor * 3, 300)
        if rms > threshold:
            voiced += frame_s
            silence = 0.0
            speaking = speaking or voiced >= settings.min_speech_s
        else:
            if not speaking:
                rms_floor = 0.9 * rms_floor + 0.1 * rms
                voiced = max(0.0, voiced - frame_s)  # isolated noises don't add up
            silence += frame_s
        elapsed = time.monotonic() - started
        if not speaking and elapsed > timeout:
            return None
        if speaking and silence >= settings.silence_s:
            break
        if elapsed > settings.max_utterance_s:
            break
    return np.concatenate(frames)


def _fold(word: str) -> str:
    """Lowercase, no accents, letters only: "Prométhée," -> "promethee"."""
    word = unicodedata.normalize("NFKD", word.lower())
    return re.sub(r"[^a-z]", "", word)


def strip_wake_word(text: str, label: str) -> tuple[str, bool]:
    """Removes the wake word Whisper may have transcribed at the start ("Hey Jarvis, quelle...").

    Returns the rest of the sentence and whether the wake word was there. Only the last word of
    the label is matched ("jarvis" for hey_jarvis), since Whisper in French mangles "hey".
    """
    key = _fold(label.replace("_", " ").split()[-1])[:5]
    words = text.split()
    for i, word in enumerate(words[:4]):
        if key and _fold(word).startswith(key):
            rest = " ".join(words[i + 1:]).lstrip(" ,.!?;:")
            return rest[:1].upper() + rest[1:], True
    return text, False


class SpeechToText:
    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel

        self.language = settings.whisper_language
        self.model = WhisperModel(
            settings.whisper_model, device="cpu", compute_type=settings.whisper_compute,
            cpu_threads=settings.whisper_threads,
        )

    def transcribe(self, audio_int16: np.ndarray) -> str:
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio, language=self.language, beam_size=1, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class TextToSpeech:
    def __init__(self, settings: Settings):
        from piper import PiperVoice

        path = Path(settings.piper_voice)
        if not path.exists():
            raise FileNotFoundError(f"Piper voice not found: {path} (run `promethee setup`)")
        self.voice = PiperVoice.load(path)
        self.rate = self.voice.config.sample_rate
        self.output_device = _device(settings.output_device)

    def synthesize(self, text: str) -> np.ndarray:
        return np.concatenate([c.audio_int16_array for c in self.voice.synthesize(text)])

    def speak(self, text: str, turn: Turn | None = None) -> None:
        """Synthesizes sentence by sentence and starts speaking after the first one: less waiting."""
        chunks: queue.Queue[np.ndarray | None] = queue.Queue()
        t0 = time.perf_counter()

        def produce():
            for c in self.voice.synthesize(text):
                chunks.put(c.audio_int16_array)
            chunks.put(None)

        threading.Thread(target=produce, daemon=True).start()
        first = True
        with sd.OutputStream(samplerate=self.rate, channels=1, dtype="int16",
                             device=self.output_device) as out:
            while (chunk := chunks.get()) is not None:
                if first and turn:
                    turn.times["tts_first_audio"] = time.perf_counter() - t0
                    first = False
                out.write(chunk.reshape(-1, 1))
        if turn:
            turn.times["tts_total"] = time.perf_counter() - t0


def resample(audio: np.ndarray, src: int, dst: int = RATE) -> np.ndarray:
    """Linear resampling, good enough for the benchmark."""
    if src == dst:
        return audio
    n = int(len(audio) * dst / src)
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.int16)
