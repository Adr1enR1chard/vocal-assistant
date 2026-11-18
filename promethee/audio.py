"""Ear and voice: microphone, wake word, recording until silence, STT and TTS."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import Settings
from .metrics import Turn

RATE = 16_000
FRAME = 1280  # 80 ms: the block size openWakeWord expects


def _device(value: str | None):
    return int(value) if value and value.isdigit() else value


class Microphone:
    """Continuous microphone stream (16 kHz mono int16) split into 80 ms blocks."""

    def __init__(self, settings: Settings):
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
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
        return self.frames.get()


class WakeWord:
    def __init__(self, settings: Settings):
        from openwakeword.model import Model

        if settings.wake_word_is_custom and not Path(settings.wake_word).exists():
            raise FileNotFoundError(
                f"Wake word model not found: {settings.wake_word}. Train it first (see README, "
                "\"Wake word\"), use `--ptt`, or set WAKE_WORD=hey_jarvis meanwhile."
            )
        self.threshold = settings.wake_threshold
        self.model = Model(wakeword_models=[settings.wake_word], inference_framework="onnx")

    def wait(self, mic: Microphone) -> float:
        mic.flush()
        self.model.reset()
        while True:
            scores = self.model.predict(mic.read())
            score = max(scores.values())
            if score >= self.threshold:
                return score


def record_utterance(mic: Microphone, settings: Settings) -> np.ndarray | None:
    """Records until the user stops talking. Energy threshold adapts to the ambient noise."""
    frames: list[np.ndarray] = []
    rms_floor = None
    speaking = False
    silence = 0.0
    started = time.monotonic()
    frame_s = FRAME / RATE
    while True:
        frame = mic.read()
        frames.append(frame)
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        if rms_floor is None:
            rms_floor = rms
        threshold = max(rms_floor * 3, 300)
        if rms > threshold:
            speaking, silence = True, 0.0
        else:
            rms_floor = 0.9 * rms_floor + 0.1 * rms if not speaking else rms_floor
            silence += frame_s
        elapsed = time.monotonic() - started
        if not speaking and elapsed > settings.no_speech_timeout_s:
            return None
        if speaking and silence >= settings.silence_s:
            break
        if elapsed > settings.max_utterance_s:
            break
    return np.concatenate(frames)


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
