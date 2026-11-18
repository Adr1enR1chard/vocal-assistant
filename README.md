# Prométhée

A 100% local voice assistant that can call MCP tools, designed to run on a cheap box. Say "Prométhée", then ask your question.

The assistant speaks French for now: the system prompt, the tool descriptions, Whisper and the Piper voice are all set up for French. The code and the docs are in English.

```
mic ─► openWakeWord ─► record until silence ─► faster-whisper ─► Ollama (Qwen3 4B) ⇄ MCP tools ─► Piper ─► speaker
```

| Role | Component | File |
|---|---|---|
| Brain | Ollama + `qwen3.5:4b` (optional thinking with `LLM_THINK`), forced on CPU | `promethee/agent.py` |
| Ear | faster-whisper `small`, int8, French | `promethee/audio.py` |
| Voice | Piper `fr_FR-siwis-medium`, plays as soon as the first sentence is ready | `promethee/audio.py` |
| Wake word | openWakeWord, custom "Prométhée" model | `promethee/audio.py` |
| Tools | MCP client (stdio or HTTP, OAuth handled) | `promethee/mcp_hub.py` |
| Internet | DuckDuckGo search (no key) and page reading | `promethee/web.py` |
| Metrics | time per step, tokens/s, peak RAM, CPU → `metrics.csv` | `promethee/metrics.py` |

## Install (PC)

```bash
uv sync
cp .env.example .env
uv run promethee setup          # Piper voice, wake word feature models, Whisper
```

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull qwen3.5:4b
```

## Connect Stockpot (or another MCP server)

```bash
cp mcp_servers.example.json mcp_servers.json   # then set your Worker URL
uv run promethee login
```

The first time, the browser opens Stockpot's authorization page (password `APP_PASSWORD`). Tokens are kept in `.tokens/` and refreshed automatically. On the box, just copy that folder: no browser needed.

Server instructions (MCP `instructions`) are appended to the system prompt. Several servers can coexist, over HTTP or stdio (see the header of `promethee/mcp_hub.py`).

## Web search

The assistant has two built-in tools, enabled by default (`WEB_SEARCH=1`): `web_search` (DuckDuckGo, no key, with a news mode) and `read_page`, which opens a result and keeps only the passages related to the question. It uses them for anything that changes or depends on the location: stocks, weather, cinema showtimes, news…

Put your city in `HOME_LOCATION` (for example `Lyon 7e`) so that "the cinema nearby" or "the weather" search in the right place. A web question often costs three or four LLM rounds: expect more latency than for an inventory question.

## Usage

```bash
uv run promethee chat            # keyboard: tests the brain and the tools without audio
uv run promethee chat --speak    # same, with the voice
uv run promethee run --ptt       # voice, press Enter to talk
uv run promethee run             # full voice, woken up by "Prométhée"
uv run promethee bench           # benchmark without a mic, read-only on Stockpot
```

`bench` has Piper speak sentences, Whisper transcribe them, the LLM and its tools answer, then synthesizes the answer. Results are reproducible, which makes it easy to compare settings. You can pass your own sentences: `uv run promethee bench "Qu'est-ce qu'il reste comme lait ?"`.

## Measuring to pick the box

Each turn prints and writes to `metrics.csv`:

- the time of each step (STT, LLM, tools, TTS first audio);
- the **perceived latency**: from the end of your sentence to the first sound of the answer;
- the LLM speed: prompt processing and generation, in tokens/s;
- the peak RAM of the assistant and of Ollama, and the CPU load.

To simulate a box, keep `LLM_CPU_ONLY=1` and set `LLM_THREADS` and `WHISPER_THREADS` to its core count (4 for a Raspberry Pi 5 or an Intel N100 mini-PC). Your PC is still faster per core than a box: treat these numbers as a lower bound, and compare configurations against each other (`LLM_MODEL=qwen3.5:2b`, `LLM_THINK=0`, `WHISPER_MODEL=base`…).

The heaviest part will be prompt processing by the LLM: the system prompt and the descriptions of the 8 tools are over a thousand tokens, re-read on every round. A request with a tool takes at least two rounds.

## Wake word

The wake word is "Prométhée". openWakeWord only ships English models, so it needs a custom model:

1. Open openWakeWord's [automatic model training notebook](https://github.com/dscripka/openWakeWord#training-new-models) (Google Colab).
2. Set the target phrase to a phonetic English spelling that sounds like the French word, e.g. `pro may tay` (the notebook generates its synthetic clips with an English TTS). Listen to the generated samples and adjust the spelling until it sounds right.
3. Train, download the `.onnx` file and save it as `models/promethee.onnx` (the default `WAKE_WORD`).
4. Tune `WAKE_THRESHOLD` if it triggers too often (raise it) or misses you (lower it).

Until the model exists, `promethee run --ptt` works without a wake word, or set `WAKE_WORD=hey_jarvis` to use a built-in model.

## On the box (Linux)

```bash
sudo apt install libportaudio2
curl -fsSL https://ollama.com/install.sh | sh
uv sync && uv run promethee setup && ollama pull qwen3.5:4b
```

Copy `.env`, `mcp_servers.json`, `.tokens/` and `models/promethee.onnx` from the PC. `python -m sounddevice` lists microphones and speakers, to put in `INPUT_DEVICE` and `OUTPUT_DEVICE`.
