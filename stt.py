"""Speech-to-text via ElevenLabs Scribe v2 Realtime WebSocket.

Port of pkg/stt/elevenlabs.go. Streams PCM int16 audio to ElevenLabs and
returns partial/committed transcripts.
"""

import base64
import json
import logging
import queue
import threading
from typing import NamedTuple

from websockets.sync.client import connect

logger = logging.getLogger(__name__)


class Result(NamedTuple):
    text: str
    is_final: bool


class ScribeTranscriber:
    """Streams PCM audio to ElevenLabs and yields transcription results."""

    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        vad_silence_seconds: float = 0.8,
        language: str = "en",
    ):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.vad_silence_seconds = vad_silence_seconds
        self.language = language

        self._ws = None
        self._results: "queue.Queue[Result]" = queue.Queue()
        self._ready = threading.Event()
        self._reader_thread = None
        self._closed = False

    def start(self):
        """Open the WebSocket and spawn the reader thread."""
        url = (
            "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
            f"?model_id=scribe_v2_realtime"
            f"&audio_format=pcm_{self.sample_rate}"
            f"&language_code={self.language}"
            f"&commit_strategy=vad"
            f"&vad_silence_threshold_secs={self.vad_silence_seconds}"
        )
        # Auth is an HTTP header, not a subprotocol.
        self._ws = connect(url, additional_headers={"xi-api-key": self.api_key})
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="stt-reader", daemon=True
        )
        self._reader_thread.start()

    def write(self, pcm: bytes):
        """Send a PCM int16 chunk. Blocks until session_started."""
        if self._ws is None:
            raise RuntimeError("transcriber not started")
        self._ready.wait()
        encoded = base64.b64encode(pcm).decode()
        msg = {"message_type": "input_audio_chunk", "audio_base_64": encoded}
        self._ws.send(json.dumps(msg))

    def results(self) -> "queue.Queue[Result]":
        return self._results

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _read_loop(self):
        while not self._closed:
            try:
                msg = self._ws.recv()
            except Exception as e:
                logger.warning("stt read error: %s", e)
                return
            if msg is None:
                continue
            try:
                event = json.loads(msg)
            except json.JSONDecodeError:
                logger.warning("stt parse error: %s", msg)
                continue

            # The API uses "message_type" in some events, "type" in others.
            event_type = event.get("message_type") or event.get("type")
            text = event.get("text", "")

            if event_type == "session_started":
                logger.info("stt: session started")
                self._ready.set()
            elif event_type == "partial_transcript":
                if text:
                    self._results.put(Result(text=text, is_final=False))
            elif event_type == "committed_transcript":
                if text:
                    self._results.put(Result(text=text, is_final=True))
            elif event_type in (
                "error",
                "auth_error",
                "quota_exceeded",
                "transcriber_error",
                "input_error",
                "rate_limited",
                "queue_overflow",
                "resource_exhausted",
                "session_time_limit_exceeded",
                "chunk_size_exceeded",
                "insufficient_audio_activity",
            ):
                err = event.get("error") or event.get("message") or event_type
                logger.warning("stt server error: %s", err)
            else:
                logger.debug("stt unknown event: %s", msg)
