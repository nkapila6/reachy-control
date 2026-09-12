#!/usr/bin/env python3
"""Speech-driven motion generator for Reachy Mini.

This is a Python port of the Go motion model in reachy-utils/pkg/motion. It
produces head pose, antenna angles and body yaw at 50 Hz, driven by a speech
level signal. The Pose is sent to the robot over a WebSocket as a
set_full_target command.
"""

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass

from websockets.sync.client import connect

logger = logging.getLogger(__name__)

# Unit conversion.
d2r = math.pi / 180

# Breathing motion.
breathe_z = 0.005
breathe_hz = 0.1

# Antenna sway parameters.
ant_sway_rad = 15 * d2r
ant_hz = 0.5
ant_rest_r = -0.1745
ant_rest_l = 0.1745

# Body yaw sway parameters.
body_yaw_rad = 2.5 * d2r
body_yaw_hz = 0.06

# Spring constants for antenna physics.
ant_stiff = 280.0
ant_damp = 13.0

# Head sway: (amplitude, frequency) per axis. Frequencies are chosen to avoid
# obvious synchronisation so the motion looks organic.
sway_pitch = (4.5 * d2r, 2.2)
sway_yaw = (7.5 * d2r, 0.6)
sway_roll = (2.25 * d2r, 1.3)
sway_x = (0.0045, 0.35)
sway_y = (0.00375, 0.45)
sway_z = (0.00225, 0.25)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _se3(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> list[float]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        x,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        y,
        -sp,
        cp * sr,
        cp * cr,
        z,
        0,
        0,
        0,
        1,
    ]


def _spring(pos: float, vel: float, target: float, dt: float) -> tuple[float, float]:
    vel += ((target - pos) * ant_stiff - vel * ant_damp) * dt
    return pos + vel * dt, vel


@dataclass
class Pose:
    """Output of the motion model."""

    head: list[float]
    antennas: list[float]
    body_yaw: float


class MotionModel:
    """Produces a continuous, speech-driven Pose at 50 Hz."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target = 0.0

        now = time.time()
        self._start = now
        self._last = now

        self._env = 0.0
        self._loud = 0.0

        # Random phases keep each robot's motion from looking identical.
        self._ph_pitch = random.random() * 2 * math.pi
        self._ph_yaw = random.random() * 2 * math.pi
        self._ph_roll = random.random() * 2 * math.pi
        self._ph_x = random.random() * 2 * math.pi
        self._ph_y = random.random() * 2 * math.pi
        self._ph_z = random.random() * 2 * math.pi

        self._ant_pos_r = ant_rest_r
        self._ant_vel_r = 0.0
        self._ant_pos_l = ant_rest_l
        self._ant_vel_l = 0.0

    def set_level(self, v: float) -> None:
        """Set the speech level (0..1). Higher values make the robot more animated."""
        with self._lock:
            self._target = _clamp(v, 0.0, 1.0)

    def _get_target(self) -> float:
        with self._lock:
            return self._target

    def tick(self) -> Pose:
        """Advance the model and return the next Pose. Call at ~50 Hz."""
        now = time.time()
        dt = min(now - self._last, 0.05)
        self._last = now
        t = now - self._start
        lv = self._get_target()

        # Envelope follower: fast attack when speech appears, slower release.
        env_target, env_tau = 0.0, 0.28
        if lv > 0.05:
            env_target = 1.0
        if env_target > self._env:
            env_tau = 0.05
        self._env += (env_target - self._env) * (1 - math.exp(-dt / env_tau))

        # Loudness follower tracks the raw speech level.
        self._loud += (lv - self._loud) * (1 - math.exp(-dt / 0.06))

        # g is the overall animation gain, capped so things never get too wild.
        g = _clamp(math.pow(self._loud, 0.7) * 2.3, 0.0, 1.25) * self._env

        def osc(a: tuple[float, float], ph: float) -> float:
            return a[0] * g * math.sin(2 * math.pi * a[1] * t + ph)

        pitch = osc(sway_pitch, self._ph_pitch)
        yaw = osc(sway_yaw, self._ph_yaw)
        roll = osc(sway_roll, self._ph_roll)
        x = osc(sway_x, self._ph_x)
        y = osc(sway_y, self._ph_y)
        z = breathe_z * math.sin(2 * math.pi * breathe_hz * t) + osc(sway_z, self._ph_z)

        # Antennas sway in opposition and perk up with speech energy.
        ant_b = (
            ant_sway_rad * math.sin(2 * math.pi * ant_hz * t) * (0.3 + 0.7 * self._env)
        )
        perk = 5 * d2r * g
        self._ant_pos_r, self._ant_vel_r = _spring(
            self._ant_pos_r, self._ant_vel_r, ant_rest_r - ant_b - perk, dt
        )
        self._ant_pos_l, self._ant_vel_l = _spring(
            self._ant_pos_l, self._ant_vel_l, ant_rest_l + ant_b - perk, dt
        )

        body_yaw = (
            body_yaw_rad
            * math.sin(2 * math.pi * body_yaw_hz * t)
            * (1 - 0.5 * self._env)
        )

        return Pose(
            head=_se3(x, y, z, roll, pitch, yaw),
            antennas=[self._ant_pos_r, self._ant_pos_l],
            body_yaw=body_yaw,
        )


class MotionController:
    """Runs the motion model and streams poses to the robot over WebSocket."""

    def __init__(self, robot_host: str = "localhost", port: int = 8000) -> None:
        self.robot_host = robot_host
        self.port = port
        self.model = MotionModel()

        self._ws = None
        self._stop_event = threading.Event()
        self._motion_on = False
        self._thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        """Connect to the robot and start the 50 Hz motion loop."""
        uri = f"ws://{self.robot_host}:{self.port}/ws/sdk"
        logger.info("connecting to motion WebSocket at %s", uri)
        self._ws = connect(uri)
        logger.info("motion WebSocket connected")

        self._stop_event.clear()
        self._motion_on = True

        self._reader_thread = threading.Thread(
            target=self._drain_incoming, name="motion-ws-reader", daemon=True
        )
        self._reader_thread.start()

        self._thread = threading.Thread(
            target=self._tick_loop, name="motion-tick", daemon=True
        )
        self._thread.start()

    def set_speaking(self, speaking: bool) -> None:
        """Toggle between speaking and idle motion.

        When True: level 0.7 (active head sway, antenna perk).
        When False: level 0.15 (gentle idle sway).
        The motion loop only sends poses when _motion_on is True.
        """
        self._motion_on = True
        level = 0.7 if speaking else 0.15
        self.model.set_level(level)

    def stop_motion(self) -> None:
        """Stop sending poses entirely (agent is done responding)."""
        self._motion_on = False
        self.model.set_level(0.0)

    def stop(self) -> None:
        """Send goto_sleep, stop the loop and close the WebSocket."""
        logger.info("stopping motion controller")
        self._motion_on = False
        self.model.set_level(0.0)

        if self._ws is not None:
            try:
                self._ws.send(json.dumps({"type": "goto_sleep"}))
            except Exception as e:
                logger.warning("goto_sleep failed: %s", e)
            time.sleep(1.5)

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

        if self._ws is not None:
            try:
                self._ws.close()
            except Exception as e:
                logger.warning("websocket close failed: %s", e)
            self._ws = None

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1)

    def _drain_incoming(self) -> None:
        # Drop anything the robot sends back so the socket buffers don't fill.
        while not self._stop_event.is_set():
            try:
                self._ws.recv()
            except Exception:
                break

    def _tick_loop(self) -> None:
        period = 0.02  # 50 Hz
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            if self._motion_on and self._ws is not None:
                try:
                    pose = self.model.tick()
                    message = {
                        "type": "set_full_target",
                        "head": pose.head,
                        "antennas": pose.antennas,
                        "body_yaw": pose.body_yaw,
                    }
                    self._ws.send(json.dumps(message))
                except Exception as e:
                    logger.warning("motion tick failed: %s", e)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))
