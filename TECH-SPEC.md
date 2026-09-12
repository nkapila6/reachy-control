# TECH-SPEC — reachy-control

Architecture and data flow for the local Reachy Mini voice agent.

## Component map

| Module | Role |
| --- | --- |
| `main.py` | Entry point, wiring, conversation loop, history, barge-in. |
| `stt.py` | ElevenLabs Scribe v2 Realtime WebSocket client. |
| `llm.py` | OpenAI-compatible streaming chat completions + tool call accumulation. |
| `tts.py` | OpenRouter TTS (default) and ElevenLabs TTS (alternate). |
| `reachy_audio.py` | Mic capture, playback queue, resampling, volume boost, speaking signal. |
| `motion.py` | 50 Hz motion loop over `ws://host:port/ws/sdk`. |
| `agent_tools.py` | Tool registry; dynamic spec inclusion based on env/flags. |
| `context_tools.py` | Exa `/search`, `search_and_summarize`, `fetch_and_summarize`, `llm_context`. |
| `pinchtab_tools.py` | Plain functions that call a PinchTab HTTP browser API. |


## Data flow

```
Reachy mic
    |
    v
reachy_audio._mic_loop: normalize to mono int16 PCM 16 kHz
    |
    v
stt.ScribeTranscriber: base64 input_audio_chunk over WSS
    |
    v
partial_transcript (is_final=False)  -> logged as interim
committed_transcript (is_final=True) -> main.py _listen()
    |
    v
main.py _respond_and_speak()
    |\
    | `- history (last 20 messages) + user utterance -> messages
    |
    v
llm.stream_chat(messages, tools=tool_specs)
    |\
    | `- SSE deltas accumulate content tokens and tool_calls fragments
    |
    |-- content token --------------------------------------> sentence buffer
    |                                                          |
    |                                                          v
    |                                                flush on last .!? in buffer
    |                                                          |
    |                                                          v
    |                                                 tts.synthesize(sentence)
    |                                                          |
    |                                                          v
    |                                              reachy_audio.output(pcm, sr)
    |                                                          |
    |                                                          v
    |                                               _output_loop decimates,
    |                                               boosts, pushes to speaker
    |
    `-- tool_call finish -----------------------------------> execute_tool()
                                                              |
                                                              v
                                                agent_tools -> context_tools
                                                         -> pinchtab_tools
                                                              |
                                                              v
                                                append assistant + tool messages
                                                              |
                                                              v
                                                next LLM round (max 3 rounds)
```

Barge-in during playback: `_speak_sentence` estimates playback duration, polls the STT result queue, and on any final transcript calls `audio.interrupt()` and returns `interrupted=True`. The rest of the response is abandoned and history is not saved.

## Audio format contract

| Stage | Format | Notes |
| --- | --- | --- |
| Mic frames from robot | multi-channel or float | Normalized to mono int16 in `_mic_loop`. |
| STT input | mono int16 PCM 16 kHz | Sent as base64 `input_audio_chunk` messages. |
| OpenRouter TTS output | PCM 24 kHz int16 | Needs index decimation to 16 kHz. |
| ElevenLabs TTS output | PCM 16 kHz int16 (configurable) | No resampling needed. |
| Speaker input | float32 [-1, 1] | `int16 / 32768`, optionally decimated, boosted 5.0, clipped. |

Resampling is simple index decimation (`np.arange(0, len(audio), ratio)`), not a proper low-pass resampler. It is good enough for voice TTS but leaves some aliasing.

## STT protocol

WebSocket URL:
```
wss://api.elevenlabs.io/v1/speech-to-text/realtime
  ?model_id=scribe_v2_realtime
  &audio_format=pcm_16000
  &language_code={language}
  &commit_strategy=vad
  &vad_silence_threshold_secs=0.8
```

Auth header: `xi-api-key: {ELEVENLABS_API_KEY}`.

Client sends: `{"message_type": "input_audio_chunk", "audio_base_64": "..."}`.

Server events consumed:
- `session_started` — unblocks `write()`.
- `partial_transcript` — non-final text.
- `committed_transcript` — final text, VAD commit.
- Error variants are logged and ignored.

## LLM SSE and tool calling

`llm.py` posts to `{base_url}/chat/completions` with `stream: true`.
It yields:
- `{"type": "token", "content": str}` for each content delta.
- `{"type": "tool_call", "tool_call": {"id", "name", "arguments": dict}}` after `finish_reason == "tool_calls"` or if pending calls remain at stream end.

`main.py` accumulates content into a sentence buffer and appends `assistant` + `tool` messages for each tool-calling round, up to `MAX_TOOL_ROUNDS = 3`.

## Sentence flush rule

The buffer is scanned for `.`, `!`, or `?`. When any is present, the sentence is cut at the last such character. Remaining text stays in the buffer for the next token. This keeps TTS latency low without waiting for the full response.

## Barge-in semantics

Only final transcripts interrupt. `_speak_sentence` waits out the estimated playback duration while polling. A final transcript during that window:
- calls `audio.interrupt()` to clear the queue and player,
- returns `True`,
- causes `_respond_and_speak` to break out of all rounds,
- and prevents history from being saved for that turn.

If the user is silent, the remaining queued sentences finish.

## Motion levels

`MotionController.set_speaking(True)` sets level 0.7; `False` sets 0.15. `goto_sleep` is sent on shutdown. The motion thread sends `{"type": "set_full_target", "head", "antennas", "body_yaw"}` at 50 Hz.

## Tool registry

`agent_tools.init_agent_tools` builds the OpenAI tool spec list dynamically:
- `web_search` is included only if `EXA_API_KEY` is set.
- PinchTab tools are included only if `--pinchtab-url` is set.
- `--no-tools` passes an empty spec list.

Tool failures return JSON `{"error": str(e)}` so the model can adjust.

## Exa integration shapes

`context_tools.py` uses `requests` against `https://api.exa.ai`.

`web_search`:
```json
POST /search
{
  "query": "...",
  "numResults": 10,
  "contents": {"text": {"maxCharacters": 2000}}
}
```

Returns `title`, `url`, `description`, `content` per result.

## Environment variables

| Variable | Required | Used by |
| --- | --- | --- |
| `ELEVENLABS_API_KEY` | yes | STT, optional ElevenLabs TTS |
| `OPENROUTER_API_KEY` | yes | LLM, default TTS |
| `EXA_API_KEY` | no | `web_search` tool |
| `REACHY_HOST` | yes | robot daemon |
| `REACHY_PORT` | yes | robot daemon |
| `PINCHTAB_URL` | no | browser tools |
| `PINCHTAB_TOKEN` | no | browser tools |

## Deploy flow

`deploy.sh`:
1. Loads SSH key if not already present.
2. Prompts for `.env` values if missing or if `--env` is passed.
3. SSHes to the robot to ensure the daemon Python 3.12 exists and sets speaker volume.
4. SHA256-compares local files to remote files in `~/reachy-control`.
5. Stops any running `python.*main.py` if changes need copying.
6. `scp`s changed files.
7. Runs `/opt/uv/uv sync` on the robot.

## What was deliberately not ported

From the Go `reachy-utils` implementation:
- The subprocess bridge — this Python port runs in a single process.
- The Deepgram STT stub.
- The dead Kokoro local TTS path.

Everything else (Scribe v2, OpenRouter streaming, OpenRouter/ElevenLabs TTS, sentence buffering, barge-in, motion model, tool set) is ported.
