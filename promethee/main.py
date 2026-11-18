"""Prométhée, a local voice assistant.

  promethee setup          downloads the Piper voice, the wake word feature models and the Whisper model
  promethee login          connects to the MCP servers (OAuth the first time) and lists the tools
  promethee chat           keyboard conversation (tests the brain + MCP without audio)
  promethee run            full voice loop, woken up by "Prométhée"
  promethee run --ptt      same, but press Enter instead of saying the wake word
  promethee bench          reproducible benchmark, no microphone (synthetic voice → STT → LLM → TTS)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from .aio import in_thread
from .config import ROOT, Settings
from .metrics import MetricsLog, Turn

# Spoken to the (French) agent, read-only on Stockpot
BENCH_PHRASES = [
    "Quel jour sommes-nous ?",
    "Qu'est-ce qu'il y a dans le frigo ?",
    "Qu'est-ce qui va bientôt périmer ?",
    "Qu'est-ce qu'il y a sur la liste de courses ?",
    "Est-ce qu'il me reste des yaourts ?",
]


def _metrics(s: Settings) -> MetricsLog:
    return MetricsLog(s.metrics_csv, {
        "llm_model": s.llm_model, "whisper_model": s.whisper_model, "llm_threads": s.llm_threads,
    })


def cmd_setup(s: Settings) -> None:
    voice = s.piper_voice.stem
    print(f"⬇️  Piper voice {voice}")
    subprocess.run([sys.executable, "-m", "piper.download_voices", voice,
                    "--data-dir", str(s.piper_voice.parent)], check=True)
    print(f"⬇️  Wake word models ({s.wake_word_label})")
    from openwakeword.utils import download_models
    # Always fetches the shared feature models; a built-in wake word is fetched by name.
    download_models([s.wake_word_label])
    if s.wake_word_is_custom and not Path(s.wake_word).exists():
        print(f"⚠️  {s.wake_word} is missing: train it (see README, \"Wake word\"). "
              "Until then, use `promethee run --ptt` or WAKE_WORD=hey_jarvis.")
    print(f"⬇️  Whisper {s.whisper_model}")
    from faster_whisper import WhisperModel
    WhisperModel(s.whisper_model, device="cpu", compute_type=s.whisper_compute)
    print(f"✅ Ready. Don't forget `ollama pull {s.llm_model}`.")


async def cmd_login(s: Settings) -> None:
    from .mcp_hub import McpHub
    async with McpHub(s) as hub:
        print(f"✅ {len(hub.ollama_tools)} tools available. Tokens saved in {s.token_dir}")


async def cmd_chat(s: Settings, speak: bool) -> None:
    from .agent import Agent
    from .mcp_hub import McpHub

    tts = None
    if speak:
        from .audio import TextToSpeech
        tts = TextToSpeech(s)
    metrics = _metrics(s)
    async with McpHub(s) as hub:
        agent = Agent(s, hub)
        print(f"🧠 Loading {s.llm_model}…")
        await agent.warmup()
        print("💬 Type your request (empty to quit).")
        while (text := (await in_thread(input, "you > ")).strip()):
            turn = Turn(heard=text)
            turn.sampler.start()
            t0 = time.perf_counter()
            turn.answer = await agent.ask(text, turn)
            print(f"🤖 {turn.answer}")
            if tts:
                await in_thread(tts.speak, turn.answer, turn)
            turn.times["total"] = time.perf_counter() - t0
            turn.sampler.stop()
            print(turn.summary())
            metrics.write(turn)


async def cmd_run(s: Settings, ptt: bool) -> None:
    from .agent import Agent
    from .audio import (Microphone, SpeechToText, TextToSpeech, WakeWord, record_utterance,
                        strip_wake_word)
    from .mcp_hub import McpHub

    print("🧩 Loading audio models…")
    wake = None if ptt else WakeWord(s)
    stt, tts = SpeechToText(s), TextToSpeech(s)
    metrics = _metrics(s)
    async with McpHub(s) as hub:
        agent = Agent(s, hub)
        print(f"🧠 Loading {s.llm_model}…")
        await agent.warmup()
        with Microphone(s) as mic:
            # "wake": wait for the wake word. "follow_up": the assistant just answered, listen a
            # while without the wake word. "awake": only the wake word was heard, listen again.
            state = "wake"
            while True:
                preroll = None
                timeout = s.no_speech_timeout_s
                if ptt:
                    await in_thread(input, "\n⏎  Press Enter, then speak…")
                    mic.flush()
                elif state == "follow_up":
                    await asyncio.sleep(0.2)  # let the speaker's echo die out
                    mic.flush()
                    timeout = s.follow_up_s
                    print(f"\n👂 Still listening for {s.follow_up_s:g}s…")
                elif state == "awake":
                    print("👂 Listening.")
                else:
                    print(f'\n👂 Listening for "{s.wake_word_label}"…')
                    score, preroll = await in_thread(wake.wait, mic)
                    print(f"✨ Woken up (score {score:.2f}), listening.")
                state = "wake"

                turn = Turn()
                turn.sampler.start()
                t_start = time.perf_counter()
                with turn.step("record"):
                    audio = await in_thread(record_utterance, mic, s, timeout, preroll)
                if audio is None:
                    turn.sampler.stop()
                    print("🤷 Heard nothing.")
                    continue
                with turn.step("stt"):
                    turn.heard = await in_thread(stt.transcribe, audio)
                if wake:
                    turn.heard, had_wake_word = strip_wake_word(turn.heard, s.wake_word_label)
                    if had_wake_word and not turn.heard:
                        turn.sampler.stop()
                        state = "awake"
                        continue
                print(f"🗣  {turn.heard!r}")
                if not turn.heard:
                    turn.sampler.stop()
                    continue
                turn.answer = await agent.ask(turn.heard, turn)
                print(f"🤖 {turn.answer}")
                await in_thread(tts.speak, turn.answer, turn)
                if wake and s.follow_up_s > 0:
                    state = "follow_up"
                turn.times["total"] = time.perf_counter() - t_start
                turn.sampler.stop()
                print(turn.summary())
                metrics.write(turn)


async def cmd_bench(s: Settings, play: bool, phrases: list[str]) -> None:
    from .agent import Agent
    from .audio import SpeechToText, TextToSpeech, resample
    from .mcp_hub import McpHub

    t = time.perf_counter()
    stt = SpeechToText(s)
    print(f"🧩 Whisper {s.whisper_model} loaded in {time.perf_counter() - t:.1f}s")
    t = time.perf_counter()
    tts = TextToSpeech(s)
    print(f"🧩 Piper loaded in {time.perf_counter() - t:.1f}s")
    metrics = _metrics(s)
    async with McpHub(s) as hub:
        agent = Agent(s, hub)
        t = time.perf_counter()
        await agent.warmup()
        print(f"🧠 {s.llm_model} loaded in {time.perf_counter() - t:.1f}s "
              f"(CPU only: {s.llm_cpu_only}, threads: {s.llm_threads or 'auto'})")
        turns: list[Turn] = []
        for phrase in phrases:
            print(f'\n🎙  "{phrase}"')
            spoken = resample(await in_thread(tts.synthesize, phrase), tts.rate)  # simulates the mic
            turn = Turn()
            turn.sampler.start()
            t0 = time.perf_counter()
            with turn.step("stt"):
                turn.heard = await in_thread(stt.transcribe, spoken)
            print(f"🗣  {turn.heard!r}")
            turn.answer = await agent.ask(turn.heard, turn)
            print(f"🤖 {turn.answer}")
            if play:
                await in_thread(tts.speak, turn.answer, turn)
            else:
                t1 = time.perf_counter()
                for i, _ in enumerate(tts.voice.synthesize(turn.answer)):
                    if i == 0:
                        turn.times["tts_first_audio"] = time.perf_counter() - t1
                turn.times["tts_total"] = time.perf_counter() - t1
            turn.times["total"] = time.perf_counter() - t0
            turn.sampler.stop()
            print(turn.summary())
            metrics.write(turn)
            turns.append(turn)

    def avg(f):
        return sum(f(t) for t in turns) / len(turns)

    print("\n📊 Averages:")
    print(f"  STT {avg(lambda t: t.times['stt']):.2f}s | LLM {avg(lambda t: t.times['llm']):.2f}s | "
          f"tools {avg(lambda t: t.times['tools']):.2f}s | TTS first audio {avg(lambda t: t.times['tts_first_audio']):.2f}s")
    print(f"  Perceived latency (end of sentence → first audio): {avg(lambda t: t.latency):.2f}s")
    print(f"  Generation {avg(lambda t: t.gen_tps):.1f} tok/s | prompt {avg(lambda t: t.prompt_tps):.0f} tok/s")
    print(f"  Peak RAM: assistant {max(t.sampler.peak_assistant_mb for t in turns):.0f} MB, "
          f"ollama {max(t.sampler.peak_ollama_mb for t in turns):.0f} MB")
    print(f"  Details in {s.metrics_csv.relative_to(ROOT)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="promethee", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("login")
    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--speak", action="store_true", help="also read the answers out loud")
    p_run = sub.add_parser("run")
    p_run.add_argument("--ptt", action="store_true", help="press Enter instead of saying the wake word")
    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--play", action="store_true", help="play the answers on the speaker")
    p_bench.add_argument("phrases", nargs="*", help="phrases to test (default: a read-only list)")
    args = parser.parse_args()

    s = Settings()
    try:
        if args.cmd == "setup":
            cmd_setup(s)
        elif args.cmd == "login":
            asyncio.run(cmd_login(s))
        elif args.cmd == "chat":
            asyncio.run(cmd_chat(s, args.speak))
        elif args.cmd == "run":
            asyncio.run(cmd_run(s, args.ptt))
        elif args.cmd == "bench":
            asyncio.run(cmd_bench(s, args.play, args.phrases or BENCH_PHRASES))
    except KeyboardInterrupt:
        print("\n👋 Bye.", flush=True)
        sys.stderr.flush()
        # MCP sessions and the microphone are already closed. A thread may still be waiting on
        # the keyboard or the mic: a normal exit would crash trying to join it, so exit hard.
        os._exit(130)


if __name__ == "__main__":
    main()
