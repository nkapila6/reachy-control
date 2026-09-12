#!/usr/bin/env python3
"""Entry point for the local Reachy voice agent.

Port of cmd/reachy-agent/main.go: local STT (ElevenLabs Scribe v2), streaming
LLM (OpenRouter), cloud TTS (OpenRouter or ElevenLabs), and speech-driven
motion. Replaces the ElevenLabs Conversational AI cloud agent.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

import agent_tools
from llm import LLMClient
from reachy_audio import ReachyAudio
from stt import ScribeTranscriber
from tts import ElevenLabsTTS, OpenRouterTTS

logger = logging.getLogger("reachy-agent")

SYSTEM_PROMPT = (
    "You are a friendly, helpful robot assistant named Reachy. You are a "
    "physical robot with a moving head and expressive antennas. Keep responses "
    "concise and conversational, 1 to 3 sentences. Be warm and engaging. You "
    "remember past context from the conversation."
)

EXIT_WORDS = {"exit", "quit", "goodbye", "bye", "stop"}

# Max tool-calling rounds per turn before giving up.
MAX_TOOL_ROUNDS = 3


def load_env():
    """Load .env file into os.environ if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value


def is_exit_command(s: str) -> bool:
    return s.lower().strip() in EXIT_WORDS


class Agent:
    def __init__(self, args):
        self.args = args
        self.audio = None
        self.motion = None
        self.transcriber = None
        self.llm = None
        self.tts = None
        self.tool_specs: list[dict] = []
        self.history: list[dict] = []

    def init(self):
        # STT key is required.
        stt_key = os.getenv("ELEVENLABS_API_KEY")
        if not stt_key:
            logger.error("ELEVENLABS_API_KEY is required (set in .env)")
            sys.exit(1)

        llm_key = self.args.llm_key or os.getenv("OPENROUTER_API_KEY")
        if not llm_key:
            logger.error("OPENROUTER_API_KEY is required (set in .env)")
            sys.exit(1)

        # Optional web + browser tools (agent_tools.py owns the registry).
        exa_key = os.getenv("EXA_API_KEY")
        if not self.args.no_tools:
            self.tool_specs = agent_tools.init_agent_tools(
                pinchtab_url=self.args.pinchtab_url,
                pinchtab_token=self.args.pinchtab_token,
                exa_key=exa_key,
            )
            logger.info("tools enabled: %d", len(self.tool_specs))

        # Motion controller (unless --no-motors).
        if not self.args.no_motors:
            from motion import MotionController

            self.motion = MotionController(
                robot_host=self.args.reachy_host, port=self.args.reachy_port
            )
            self.motion.start()
            logger.info("motion controller started")

        # Audio interface.
        on_speaking = self.motion.set_speaking if self.motion else None
        self.audio = ReachyAudio(
            robot_host=self.args.reachy_host, on_speaking_change=on_speaking
        )
        logger.info("pre-starting audio interface (daemon + media)...")
        self.audio.pre_start()
        logger.info("audio interface ready")

        # STT transcriber.
        self.transcriber = ScribeTranscriber(
            api_key=stt_key, language=self.args.language
        )
        self.transcriber.start()
        logger.info("STT session ready")

        # LLM client.
        self.llm = LLMClient(
            base_url=self.args.llm_base, api_key=llm_key, model=self.args.model
        )

        # TTS backend.
        if self.args.tts == "elevenlabs":
            self.tts = ElevenLabsTTS(api_key=stt_key, voice_id=self.args.voice)
            logger.info("TTS: elevenlabs (voice=%s)", self.args.voice)
        else:
            self.tts = OpenRouterTTS(
                api_key=llm_key, model=self.args.tts_model, voice=self.args.voice
            )
            logger.info(
                "TTS: openrouter/%s (voice=%s)", self.args.tts_model, self.args.voice
            )

    def shutdown(self):
        logger.info("shutting down...")
        if self.transcriber is not None:
            self.transcriber.close()
        if self.audio is not None:
            self.audio.stop()
        if self.motion is not None:
            self.motion.stop()

    def loop(self):
        # Feed mic audio to the transcriber.
        self.audio.start(self.transcriber.write)

        while True:
            logger.info("listening...")
            utterance = self._listen()
            if utterance == "":
                continue
            logger.info("heard: %r", utterance)
            logger.info("User: %s", utterance)

            if is_exit_command(utterance):
                logger.info("goodbye!")
                return

            self._respond_and_speak(utterance)

    def _listen(self) -> str:
        """Wait for a final transcript, 30s timeout."""
        deadline = time.time() + 30
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                logger.info("stt: timeout")
                return ""
            try:
                result = self.transcriber.results().get(timeout=remaining)
            except Exception:
                logger.info("stt: timeout")
                return ""
            if result.is_final and result.text:
                return result.text.strip()
            elif result.text:
                logger.info("stt: interim: %r", result.text)

    def _respond_and_speak(self, utterance: str):
        # Drain stale STT results before generating.
        while True:
            try:
                self.transcriber.results().get_nowait()
            except Exception:
                break

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        start = 0
        if len(self.history) > 20:
            start = len(self.history) - 20
        messages.extend(self.history[start:])
        messages.append({"role": "user", "content": utterance})

        full_response = ""
        interrupted = False
        tools = self.tool_specs or None

        for _round in range(MAX_TOOL_ROUNDS):
            sentence_buf = ""
            tool_calls = []

            try:
                events = self.llm.stream_chat(messages, tools=tools)
            except Exception as e:
                logger.error("llm: %s", e)
                break

            for event in events:
                if event["type"] == "token":
                    content = event["content"]
                    full_response += content
                    sentence_buf += content

                    s = sentence_buf
                    if any(c in s for c in ".!?"):
                        idx = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
                        sentence = s[: idx + 1]
                        sentence_buf = s[idx + 1 :]

                        text = sentence.strip()
                        if not text:
                            continue
                        logger.info("speaking: %r", text)
                        if self._speak_sentence(text):
                            interrupted = True
                            break
                elif event["type"] == "tool_call":
                    tool_calls.append(event["tool_call"])

            if interrupted:
                break

            # Flush any trailing sentence from this round.
            if sentence_buf.strip():
                text = sentence_buf.strip()
                logger.info("speaking: %r", text)
                if self._speak_sentence(text):
                    interrupted = True
                    break

            if not tool_calls:
                # No tool calls this round: the model is done.
                break

            # Execute tools and append assistant + tool messages for the next round.
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                logger.info("tool call: %s(%s)", tc["name"], tc["arguments"])
                result = agent_tools.execute_tool(tc["name"], tc["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result["content"],
                    }
                )

        # Save to history only if not interrupted and response non-empty.
        response = full_response.strip()
        if response and not interrupted:
            self.history.append({"role": "user", "content": utterance})
            self.history.append({"role": "assistant", "content": response})
        logger.info("response: %r", response)
        logger.info("Agent: %s", response)

    def _speak_sentence(self, text: str) -> bool:
        """Synthesize and play one sentence. Returns True if interrupted."""
        try:
            pcm, sr = self.tts.synthesize(text)
        except Exception as e:
            logger.error("tts: %s", e)
            print(f"Reachy: {text}")
            return False

        if not pcm:
            print(f"Reachy: {text}")
            return False

        self.audio.output(pcm, sr)

        # Estimate playback duration, min 300ms.
        dur = len(pcm) / (sr * 2) + 0.1
        if dur < 0.3:
            dur = 0.3

        # Wait for playback, polling for barge-in.
        deadline = time.time() + dur
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                result = self.transcriber.results().get(timeout=remaining)
            except Exception:
                break
            if result.is_final and result.text:
                logger.info("barge-in: %r", result.text)
                self.audio.interrupt()
                return True
        return False


def main():
    load_env()

    parser = argparse.ArgumentParser(description="Reachy local voice agent")
    parser.add_argument(
        "--reachy-host",
        default=os.getenv("REACHY_HOST", "localhost"),
        help="Reachy Mini daemon host (env: REACHY_HOST)",
    )
    parser.add_argument(
        "--reachy-port",
        type=int,
        default=int(os.getenv("REACHY_PORT", "8000")),
        help="Reachy Mini daemon port (env: REACHY_PORT)",
    )
    parser.add_argument(
        "--tts",
        default=os.getenv("TTS_BACKEND", "openrouter"),
        choices=["openrouter", "elevenlabs"],
        help="TTS backend (env: TTS_BACKEND)",
    )
    parser.add_argument(
        "--tts-model",
        default=os.getenv("TTS_MODEL", "microsoft/mai-voice-2-flash"),
        help="TTS model on OpenRouter (env: TTS_MODEL)",
    )
    parser.add_argument(
        "--voice",
        default=os.getenv("VOICE", "en-US-Harper:MAI-Voice-2"),
        help="Voice name (env: VOICE)",
    )
    parser.add_argument(
        "--llm-base",
        default=os.getenv("LLM_BASE", "https://openrouter.ai/api/v1"),
        help="LLM base URL (env: LLM_BASE)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "ibm-granite/granite-4.1-8b"),
        help="LLM model name (env: MODEL)",
    )
    parser.add_argument(
        "--llm-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="LLM API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("LANGUAGE", "en"),
        help="BCP-47 language code for STT (env: LANGUAGE)",
    )
    parser.add_argument(
        "--no-motors", action="store_true", default=False, help="Disable motor control"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        default=False,
        help="Disable tool calling (pure chat)",
    )
    parser.add_argument(
        "--pinchtab-url",
        default=os.getenv("PINCHTAB_URL"),
        help="PinchTab HTTP URL (env: PINCHTAB_URL)",
    )
    parser.add_argument(
        "--pinchtab-token",
        default=os.getenv("PINCHTAB_TOKEN"),
        help="PinchTab auth token (env: PINCHTAB_TOKEN)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s %(name)s: %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(name)s: %(message)s"
        )
    # Keep reachy_mini SDK logs quiet except for warnings.
    logging.getLogger("reachy_mini").setLevel(logging.WARNING)

    logger.info("config:")
    logger.info("  reachy_host: %s:%d", args.reachy_host, args.reachy_port)
    logger.info("  tts: %s", args.tts)
    logger.info("  tts_model: %s", args.tts_model)
    logger.info("  voice: %s", args.voice)
    logger.info("  llm_base: %s", args.llm_base)
    logger.info("  model: %s", args.model)
    logger.info("  language: %s", args.language)
    logger.info("  motors: %s", "off" if args.no_motors else "on")
    logger.info("  tools: %s", "off" if args.no_tools else "on")

    agent = Agent(args)
    agent.init()

    def shutdown(sig, frame):
        agent.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("ready -- talk to the robot!")
    try:
        agent.loop()
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
