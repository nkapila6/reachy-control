"""Reachy Mini audio backend: mic capture and speaker playback.

Plain class (no ElevenLabs SDK). Mic loop feeds PCM to the transcriber via
on_frame; output() queues TTS PCM for the playback thread.
"""

import logging
import threading
import time
from queue import Empty, Queue

import numpy as np

logger = logging.getLogger(__name__)

# DSP tuning used by the Reachy Mini audio pipeline. AGC + noise suppression
# keeps the robot mic sounding clean for the conversation model.
AUDIO_STARTUP_CONFIG = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)

# The Reachy speaker is quiet; boost incoming agent audio while staying in
# the [-1, 1] float range expected by push_audio_sample.
OUTPUT_VOLUME_BOOST = 5.0


class ReachyAudio:
    """Thread-backed mic/speaker interface for Reachy Mini hardware."""

    def __init__(self, robot_host="localhost", on_speaking_change=None):
        self.robot_host = robot_host
        self.robot = None
        self.on_frame = None
        self.input_sample_rate = 16000

        self._mic_thread = None
        self._output_thread = None
        self._output_queue = Queue()
        self._stop_event = threading.Event()

        # Motion: callback fired when TTS playback starts/stops.
        self._on_speaking_change = on_speaking_change
        self._is_speaking = False
        self._last_audio_time = 0.0

    def pre_start(self):
        """Do the slow init (daemon, ReachyMini, media, DSP) before the loop."""
        from reachy_mini import ReachyMini

        self._ensure_daemon_ready()

        # Disable automatic_body_yaw so the SDK doesn't fight our
        # MotionController for head/body control.
        logger.info("connecting to Reachy (auto-detect)...")
        self.robot = ReachyMini(automatic_body_yaw=False)

        logger.info("waking up robot...")
        try:
            self.robot.enable_motors()
            self.robot.wake_up()
        except Exception as e:
            logger.warning("wake_up failed: %s", e)

        # Disable the SDK's built-in head wobbler - our MotionController
        # handles speech-driven motion.
        try:
            self.robot.media.audio.disable_wobbling()
            logger.info("head wobbler disabled")
        except Exception as e:
            logger.warning("could not disable head wobbler: %s", e)

        logger.info("starting media pipelines...")
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        time.sleep(1)

        self._apply_audio_config()

        try:
            self.input_sample_rate = self.robot.media.get_input_audio_samplerate()
            logger.info("input sample rate: %d Hz", self.input_sample_rate)
        except Exception:
            self.input_sample_rate = 16000
            logger.info(
                "could not get sample rate, assuming %d Hz", self.input_sample_rate
            )

    def start(self, on_frame):
        """Start the mic and playback threads."""
        self.on_frame = on_frame
        self._stop_event.clear()
        self._mic_thread = threading.Thread(
            target=self._mic_loop, name="reachy-mic", daemon=True
        )
        self._output_thread = threading.Thread(
            target=self._output_loop, name="reachy-output", daemon=True
        )
        self._mic_thread.start()
        self._output_thread.start()

    def output(self, pcm: bytes, sample_rate: int):
        """Queue TTS PCM for playback. sample_rate is the PCM's source rate."""
        self._output_queue.put((pcm, sample_rate))
        # Toggle motion to "speaking" when TTS audio arrives.
        if self._on_speaking_change and not self._is_speaking:
            self._is_speaking = True
            self._on_speaking_change(True)
            logger.info("TTS audio started")
        self._last_audio_time = time.time()

    def interrupt(self):
        """Drop buffered audio and flush the hardware player (barge-in)."""
        logger.info("interrupt: clearing audio")
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except Empty:
                break

        if self._on_speaking_change and self._is_speaking:
            self._is_speaking = False
            self._on_speaking_change(False)

        if self.robot is not None:
            try:
                self.robot.media.audio.clear_player()
            except Exception as e:
                logger.warning("clear_player failed: %s", e)

    def stop(self):
        logger.info("stopping Reachy audio interface...")
        self._stop_event.set()

        current = threading.current_thread()
        if self._mic_thread is not None and current is not self._mic_thread:
            self._mic_thread.join(timeout=2)

        # Signal the output worker to exit.
        self._output_queue.put(None)
        if self._output_thread is not None and current is not self._output_thread:
            self._output_thread.join(timeout=2)

        if self.robot is not None:
            try:
                self.robot.media.stop_recording()
            except Exception as e:
                logger.warning("stop_recording failed: %s", e)
            try:
                self.robot.media.stop_playing()
            except Exception as e:
                logger.warning("stop_playing failed: %s", e)

    def _ensure_daemon_ready(self):
        """Start the daemon's hardware backend and wake the robot."""
        import urllib.request

        base = f"http://{self.robot_host}:8000"
        logger.info("starting daemon backend at %s...", base)

        # POST to /api/daemon/start?wake_up=true (GET returns 405).
        try:
            req = urllib.request.Request(
                f"{base}/api/daemon/start?wake_up=true", method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.warning("daemon start request failed: %s (continuing...)", e)

        # Poll until the backend is up.
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                resp = urllib.request.urlopen(f"{base}/api/daemon/status", timeout=5)
                body = resp.read().decode()
                if '"backend_status":null' not in body:
                    logger.info("daemon backend is up")
                    time.sleep(2)
                    return
            except Exception:
                pass
            time.sleep(1)
        logger.warning("daemon backend did not come up within 25s")

    def _apply_audio_config(self):
        audio = getattr(getattr(self.robot, "media", None), "audio", None)
        if audio is None:
            logger.warning("robot.media.audio not available")
            return

        apply_config = getattr(audio, "apply_audio_config", None)
        if not callable(apply_config):
            logger.warning("audio.apply_audio_config not available")
            return

        try:
            ok = apply_config(
                AUDIO_STARTUP_CONFIG, verify=True, write_settle_seconds=0.1
            )
            if ok:
                logger.info("audio DSP config applied")
        except Exception as e:
            logger.warning("audio DSP config error: %s", e)

    def _mic_loop(self):
        chunk_count = 0
        while not self._stop_event.is_set():
            try:
                frame = self.robot.media.get_audio_sample()
            except Exception:
                time.sleep(0.01)
                continue

            if frame is None or frame.size == 0:
                time.sleep(0.001)
                continue

            # Reachy sometimes returns multi-channel or float frames. Normalize
            # to mono int16 PCM, which is what the STT socket expects.
            if frame.ndim > 1:
                frame = frame.mean(axis=1)
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 32767.0).clip(-32768, 32767).astype(np.int16)
            elif frame.dtype != np.int16:
                frame = frame.astype(np.int16)

            if self.on_frame is not None:
                self.on_frame(frame.tobytes())

            # Log audio level every 50 chunks (~1.5s) so we can see if the
            # mic is picking up sound.
            chunk_count += 1
            if chunk_count % 50 == 1:
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                logger.info("mic: rms=%.1f chunk=%d", rms, chunk_count)

            time.sleep(0)

    def _output_loop(self):
        while True:
            item = self._output_queue.get()
            if item is None:
                break
            pcm, sample_rate = item
            self._play_pcm(pcm, sample_rate)
            # If no audio has arrived for 0.5s, ramp motion down to idle.
            if self._on_speaking_change and self._is_speaking:
                if time.time() - self._last_audio_time > 0.5:
                    self._is_speaking = False
                    self._on_speaking_change(False)

    def _play_pcm(self, pcm: bytes, sample_rate: int):
        try:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            # Resample from source rate to 16 kHz by index decimation.
            if sample_rate != 16000 and sample_rate > 0:
                ratio = float(sample_rate) / 16000.0
                indices = np.arange(0, len(audio), ratio).astype(int)
                audio = audio[indices]
            audio = np.clip(audio * OUTPUT_VOLUME_BOOST, -1.0, 1.0)
            logger.info(
                "playing TTS: %d bytes (peak=%.3f)",
                len(pcm),
                float(np.max(np.abs(audio))),
            )
            self.robot.media.push_audio_sample(audio)
        except Exception as e:
            logger.warning("playback failed: %s", e)
