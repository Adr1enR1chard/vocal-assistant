"""Per-step timing and RAM / CPU tracking during a speech turn."""

from __future__ import annotations

import csv
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import psutil

STEPS = ["record", "stt", "llm", "tools", "tts_first_audio", "tts_total", "total"]


def _ollama_processes() -> list[psutil.Process]:
    procs = []
    for p in psutil.process_iter(["name"]):
        name = (p.info["name"] or "").lower()
        if "ollama" in name or "llama-server" in name:
            procs.append(p)
    return procs


class ResourceSampler:
    """Samples RAM (assistant + Ollama) and system CPU in the background."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.me = psutil.Process()
        self.peak_assistant_mb = 0.0
        self.peak_ollama_mb = 0.0
        self.cpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _rss_mb(self, procs: list[psutil.Process]) -> float:
        total = 0
        for p in procs:
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total / 1e6

    def _run(self) -> None:
        psutil.cpu_percent(None)
        ollama = _ollama_processes()
        refresh = time.monotonic()
        while not self._stop.wait(self.interval):
            if time.monotonic() - refresh > 2:  # Ollama starts a runner on demand
                ollama, refresh = _ollama_processes(), time.monotonic()
            self.peak_assistant_mb = max(self.peak_assistant_mb, self._rss_mb([self.me]))
            self.peak_ollama_mb = max(self.peak_ollama_mb, self._rss_mb(ollama))
            self.cpu_samples.append(psutil.cpu_percent(None))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    @property
    def cpu_avg(self) -> float:
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def cpu_max(self) -> float:
        return max(self.cpu_samples, default=0.0)


@dataclass
class Turn:
    """One turn: what was heard, what was answered, and how long each step took."""

    times: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STEPS})
    heard: str = ""
    answer: str = ""
    tool_calls: list[str] = field(default_factory=list)
    llm_rounds: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    gen_seconds: float = 0.0
    prompt_seconds: float = 0.0
    sampler: ResourceSampler = field(default_factory=ResourceSampler)

    @contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - t0

    @property
    def gen_tps(self) -> float:
        return self.gen_tokens / self.gen_seconds if self.gen_seconds else 0.0

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0

    def summary(self) -> str:
        t = self.times
        return (
            f"  ⏱  listen {t['record']:.1f}s | stt {t['stt']:.2f}s | llm {t['llm']:.2f}s "
            f"({self.llm_rounds} rounds, prompt {self.prompt_tps:.0f} tok/s, generation {self.gen_tps:.1f} tok/s) | "
            f"tools {t['tools']:.2f}s | tts first audio {t['tts_first_audio']:.2f}s | "
            f"end of speech → first audio {self.latency:.2f}s\n"
            f"  💾 Peak RAM: assistant {self.sampler.peak_assistant_mb:.0f} MB, ollama {self.sampler.peak_ollama_mb:.0f} MB"
            f" | CPU avg {self.sampler.cpu_avg:.0f}% (max {self.sampler.cpu_max:.0f}%)"
        )

    @property
    def latency(self) -> float:
        """What the user feels: from the end of their sentence to the first sound of the answer."""
        t = self.times
        return t["stt"] + t["llm"] + t["tools"] + t["tts_first_audio"]


class MetricsLog:
    COLUMNS = [
        "timestamp", "heard", "answer", "tool_calls", "llm_rounds",
        *[f"{s}_s" for s in STEPS], "latency_s",
        "prompt_tokens", "gen_tokens", "prompt_tok_s", "gen_tok_s",
        "peak_assistant_mb", "peak_ollama_mb", "cpu_avg_pct", "cpu_max_pct",
        "llm_model", "whisper_model", "llm_threads",
    ]

    def __init__(self, path: Path, context: dict):
        self.path = path
        self.context = context

    def write(self, turn: Turn) -> None:
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if new:
                w.writeheader()
            w.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "heard": turn.heard,
                "answer": turn.answer,
                "tool_calls": " ".join(turn.tool_calls),
                "llm_rounds": turn.llm_rounds,
                **{f"{s}_s": round(turn.times[s], 3) for s in STEPS},
                "latency_s": round(turn.latency, 3),
                "prompt_tokens": turn.prompt_tokens,
                "gen_tokens": turn.gen_tokens,
                "prompt_tok_s": round(turn.prompt_tps, 1),
                "gen_tok_s": round(turn.gen_tps, 1),
                "peak_assistant_mb": round(turn.sampler.peak_assistant_mb),
                "peak_ollama_mb": round(turn.sampler.peak_ollama_mb),
                "cpu_avg_pct": round(turn.sampler.cpu_avg),
                "cpu_max_pct": round(turn.sampler.cpu_max),
                **self.context,
            })
