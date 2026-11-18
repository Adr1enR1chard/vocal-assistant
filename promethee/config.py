"""Settings, read from environment variables (and a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "oui"}


def _wake_word(value: str) -> str:
    """A path to a custom .onnx/.tflite model (relative to the project root), or a built-in name."""
    if value.endswith((".onnx", ".tflite")):
        return str(ROOT / value)
    return value


@dataclass
class Settings:
    # Brain (Ollama)
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.5:4b"))
    # Think before answering: chains tools better, but costs tokens on every round
    llm_think: bool = field(default_factory=lambda: _bool("LLM_THINK", True))
    llm_cpu_only: bool = field(default_factory=lambda: _bool("LLM_CPU_ONLY", True))
    llm_threads: int = field(default_factory=lambda: _int("LLM_THREADS", 4))  # 0 = let Ollama decide
    # A web question piles up search results and pages over several rounds: 8192 overflows
    llm_ctx: int = field(default_factory=lambda: _int("LLM_CTX", 16384))
    llm_max_rounds: int = field(default_factory=lambda: _int("LLM_MAX_ROUNDS", 5))
    history_turns: int = field(default_factory=lambda: _int("HISTORY_TURNS", 4))
    history_ttl_s: int = field(default_factory=lambda: _int("HISTORY_TTL_S", 300))

    # Web search (DuckDuckGo, no key) and page reading
    web_search: bool = field(default_factory=lambda: _bool("WEB_SEARCH", True))
    # City or neighbourhood, for local questions ("the cinema nearby", "the weather")
    home_location: str = field(default_factory=lambda: _env("HOME_LOCATION", ""))

    # Ear (faster-whisper)
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small"))
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "fr"))
    whisper_compute: str = field(default_factory=lambda: _env("WHISPER_COMPUTE", "int8"))
    whisper_threads: int = field(default_factory=lambda: _int("WHISPER_THREADS", 4))

    # Voice (Piper)
    piper_voice: Path = field(
        default_factory=lambda: ROOT / _env("PIPER_VOICE", "models/fr_FR-siwis-medium.onnx")
    )

    # Wake word (openWakeWord): "Prométhée", a custom model to train (see README)
    wake_word: str = field(default_factory=lambda: _wake_word(_env("WAKE_WORD", "models/promethee.onnx")))
    wake_threshold: float = field(default_factory=lambda: _float("WAKE_THRESHOLD", 0.5))
    # Audio kept from just before the detection, for a question chained right after the wake word
    wake_preroll_s: float = field(default_factory=lambda: _float("WAKE_PREROLL_S", 0.25))
    # After an answer, listen this long for a follow-up without the wake word (0 = off)
    follow_up_s: float = field(default_factory=lambda: _float("FOLLOW_UP_S", 10))

    # Recording
    silence_s: float = field(default_factory=lambda: _float("SILENCE_S", 0.8))
    # Voiced time needed before it counts as speech (ignores clicks and the wake word's tail)
    min_speech_s: float = field(default_factory=lambda: _float("MIN_SPEECH_S", 0.25))
    max_utterance_s: float = field(default_factory=lambda: _float("MAX_UTTERANCE_S", 15))
    no_speech_timeout_s: float = field(default_factory=lambda: _float("NO_SPEECH_TIMEOUT_S", 5))
    input_device: str | None = field(default_factory=lambda: os.environ.get("INPUT_DEVICE") or None)
    output_device: str | None = field(default_factory=lambda: os.environ.get("OUTPUT_DEVICE") or None)

    # MCP
    mcp_config: Path = field(default_factory=lambda: ROOT / _env("MCP_CONFIG", "mcp_servers.json"))
    token_dir: Path = field(default_factory=lambda: ROOT / _env("TOKEN_DIR", ".tokens"))
    oauth_callback_port: int = field(default_factory=lambda: _int("OAUTH_CALLBACK_PORT", 33418))

    # Metrics
    metrics_csv: Path = field(default_factory=lambda: ROOT / _env("METRICS_CSV", "metrics.csv"))

    @property
    def wake_word_label(self) -> str:
        """What to show the user: the model file name, or the built-in model name."""
        return Path(self.wake_word).stem if self.wake_word_is_custom else self.wake_word

    @property
    def wake_word_is_custom(self) -> bool:
        return self.wake_word.endswith((".onnx", ".tflite"))

    def llm_options(self) -> dict:
        options: dict = {"num_ctx": self.llm_ctx, "temperature": 0.3}
        if self.llm_cpu_only:
            options["num_gpu"] = 0  # no layer on the GPU: simulates the target box
        if self.llm_threads > 0:
            options["num_thread"] = self.llm_threads
        return options
