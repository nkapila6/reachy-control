"""Text-to-speech via OpenRouter and ElevenLabs.

Port of pkg/tts/openrouter.go and pkg/tts/tts.go.
"""

import logging
import re

import requests

logger = logging.getLogger(__name__)


class OpenRouterTTS:
    def __init__(
        self,
        api_key: str,
        model: str = "microsoft/mai-voice-2-flash",
        voice: str = "en-US-Harper:MAI-Voice-2",
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Return (pcm int16 bytes, sample_rate). PCM is 24 kHz."""
        body = {"input": text, "model": self.model}
        # Voice set -> request PCM. Without a voice the API returns MP3.
        if self.voice:
            body["voice"] = self.voice
            body["response_format"] = "pcm"

        resp = requests.post(
            "https://openrouter.ai/api/v1/audio/speech",
            json=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"openrouter tts: HTTP {resp.status_code}: {resp.text}")
        return resp.content, 24000


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str):
        self.api_key = api_key
        self.voice_id = voice_id
        # Voice settings match tts.go defaults.
        self.stability = 0.5
        self.similarity_boost = 0.75
        self.style = 0.0
        self.speed = 1.0

    def synthesize_pcm(self, text: str, sample_rate: int = 16000) -> tuple[bytes, int]:
        """Return (pcm int16 bytes, sample_rate)."""
        body = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "speed": self.speed,
            },
            "optimize_streaming_latency": 0,
        }
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            f"?output_format=pcm_{sample_rate}"
        )
        resp = requests.post(
            url,
            json=body,
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"elevenlabs tts: HTTP {resp.status_code}: {resp.text}")
        return resp.content, sample_rate


def split_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text into sentence-sized chunks, word-splitting over-long sentences.

    Port of SplitText from tts.go. Not used by the agent loop, kept for reuse.
    """
    if max_chars <= 0:
        max_chars = 500

    sentences = [s.strip() for s in re.split(r"[.!?\n]", text)]
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            chunks.append(buf.strip())
            buf = ""

    for s in sentences:
        if not s:
            continue
        if buf and len(buf) + len(s) + 1 > max_chars:
            flush()
        # A single sentence longer than max_chars gets word-split.
        if len(s) > max_chars and not buf:
            words = s.split()
            wbuf = ""
            for w in words:
                if wbuf and len(wbuf) + len(w) + 1 > max_chars:
                    chunks.append(wbuf.strip())
                    wbuf = ""
                wbuf = f"{wbuf} {w}".strip()
            if wbuf:
                chunks.append(wbuf)
            continue
        buf = f"{buf} {s}".strip()
    flush()
    return chunks
