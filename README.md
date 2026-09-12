# reachy-control — local voice agent for Reachy Mini

A voice-first AI agent that runs entirely on the robot. It listens through the Reachy Mini mic, transcribes with ElevenLabs Scribe, reasons with an OpenRouter LLM, speaks back via cloud TTS, and moves its head and antennas while it talks.

## What it does

1. Listens for speech.
2. Streams audio to ElevenLabs Scribe v2 Realtime for transcription.
3. Sends the final transcript to an OpenAI-compatible chat completions endpoint (default OpenRouter) with optional tool calling.
4. Streams the LLM response, synthesizes each sentence as it arrives, and plays it back.
5. Barges in: if you speak while the robot is talking, it stops, skips the rest, and starts answering the new utterance.
6. Moves the head and antennas at 50 Hz via the Reachy daemon WebSocket. Speaking motion is more animated; idle motion is gentle.

## Requirements

- Reachy Mini with its daemon running.
- SSH access to the robot as `pollen@reachy-mini.local` (override with `ROBOT=`).
- An ElevenLabs API key for Scribe STT.
- An OpenRouter API key for LLM and default TTS.
- Optional Exa API key for web search.
- Python 3.11+.

## Setup

```bash
cp .env.example .env
# Fill in: ELEVENLABS_API_KEY, OPENROUTER_API_KEY, optional EXA_API_KEY
./deploy.sh
```

`deploy.sh` prompts for required keys, SHA256-diff-copies changed files to `~/reachy-control`, and syncs the venv with `uv`.

## Run

```bash
ssh pollen@reachy-mini.local
cd ~/reachy-control
/opt/uv/uv run reachy-agent
```

Talk to the robot. Say one of `exit`, `quit`, `goodbye`, `bye`, or `stop` to end. Ctrl-C makes the robot go to sleep before exiting.

## Common flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--reachy-host` | `localhost` | Daemon host |
| `--reachy-port` | `8000` | Daemon port |
| `--tts` | `openrouter` | `openrouter` or `elevenlabs` |
| `--tts-model` | `microsoft/mai-voice-2-flash` | TTS model on OpenRouter |
| `--voice` | `en-US-Harper:MAI-Voice-2` | Voice name |
| `--llm-base` | `https://openrouter.ai/api/v1` | Chat completions base URL |
| `--model` | `ibm-granite/granite-4.1-8b` | LLM model |
| `--llm-key` | `OPENROUTER_API_KEY` | LLM/TTS API key |
| `--language` | `en` | STT language code |
| `--no-motors` | off | Disable head/antenna motion |
| `--no-tools` | off | Disable tool calling |
| `--pinchtab-url` | `PINCHTAB_URL` | PinchTab browser tool URL |
| `--pinchtab-token` | `PINCHTAB_TOKEN` | PinchTab bearer token |
| `--debug` | off | Verbose logging |

## How it works

`main.py` wires together local modules:

- `stt.py` — WebSocket to ElevenLabs Scribe v2 Realtime.
- `llm.py` — streaming OpenAI-compatible chat completions with tool call accumulation.
- `tts.py` — OpenRouter TTS (default) or ElevenLabs TTS (alternate).
- `reachy_audio.py` — mic capture, speaker playback, and barge-in handling.
- `motion.py` — 50 Hz speech-driven motion over the daemon WebSocket.
- `agent_tools.py` — tool registry for `web_search` (Exa) and optional PinchTab browser tools.

The agent streams the LLM response sentence by sentence. Each sentence is flushed on the last `.`, `!`, or `?` in the buffer. Tool calls run for up to three rounds before the final answer is spoken.

## Notes

- Tool results are returned to the LLM as strings so the model can recover from failures.
- Barge-in only reacts to a final (committed) transcript, not partial words.
- History keeps the last 20 messages and is only saved if the response completes without interruption.
- Web search needs `EXA_API_KEY`. Browser tools need `--pinchtab-url`.
