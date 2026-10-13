"""Drone-following guidance. Deliberately independent of OpenCV / YOLO.

Input : TargetObservation (normalized image coordinates) + safety flags.
Output: Guidance (suggested yaw rate, lateral and forward velocity).

SAFETY: this module only *suggests* corrections. The only interfaces provided
are GuidanceOnlyInterface and SimulatedDrone. MavlinkDroneInterface is a stub
that refuses to run. Do not connect it to real hardware until validated with
flight-control hardware, geofencing, a hardware kill switch and controlled tests.

Conventions: yaw + = clockwise (turn right), lateral + = right, forward + = ahead.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ControllerState(str, Enum):
    IDLE = "IDLE"                      # no target selected
    FOLLOWING = "FOLLOWING"
    TARGET_LOST = "TARGET_LOST"        # zero command, hover suggested
    STALE_FRAME = "STALE_FRAME"        # camera frames too old / missing
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    EMERGENCY_STOP = "EMERGENCY_STOP"  # latched until reset


@dataclass
class TargetObservation:
    cx: float          # target center x, 0..1 (0.5 = frame center)
    cy: float          # target center y, 0..1
    height: float      # bounding-box height / frame height, 0..1
    confidence: float  # 0..1


@dataclass
class Guidance:
    yaw_rate_dps: float = 0.0
    lateral_mps: float = 0.0
    forward_mps: float = 0.0
    state: ControllerState = ControllerState.IDLE
    reason: str = ""


@dataclass
class ControllerConfig:
    yaw_gain_dps: float = 60.0      # deg/s per unit of normalized x error (error is within +-0.5)
    max_yaw_dps: float = 30.0
    use_lateral: bool = False       # yaw alone usually centers the target; enable for strafing platforms
    lateral_gain: float = 1.0       # m/s per unit of x error
    max_lateral_mps: float = 1.0
    desired_height: float = 0.50    # keep target box at ~50% of frame height (distance proxy)
    forward_gain: float = 3.0       # m/s per unit of height error
    max_forward_mps: float = 1.0
    deadband_xy: float = 0.04
    deadband_h: float = 0.05
    min_confidence: float = 0.35
    max_yaw_accel: float = 90.0     # deg/s^2 slew limit
    max_lin_accel: float = 1.0      # m/s^2 slew limit
    lost_alert_s: float = 5.0       # after this, reason text recommends failsafe


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _deadband(e: float, d: float) -> float:
    return 0.0 if abs(e) < d else e - math.copysign(d, e)


def _slew(prev: float, target: float, max_rate: float, dt: float) -> float:
    step = max_rate * dt
    return prev + max(-step, min(step, target - prev))


class FollowController:
    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()
        self._estop = False
        self._manual = False
        self._lost_since: float | None = None
        self._last_t: float | None = None
        self._prev = (0.0, 0.0, 0.0)  # yaw, lateral, forward after slew limiting
        self.last = Guidance()

    # ---- operator / safety inputs -------------------------------------
    @property
    def manual(self) -> bool:
        return self._manual

    @property
    def estop(self) -> bool:
        return self._estop

    def set_manual(self, on: bool) -> None:
        self._manual = on

    def trigger_estop(self) -> None:
        self._estop = True

    def reset_estop(self) -> None:
        self._estop = False

    # ---- main update ---------------------------------------------------
    def _zero(self, state: ControllerState, reason: str) -> Guidance:
        self._prev = (0.0, 0.0, 0.0)  # safety states drop commands immediately
        self.last = Guidance(0.0, 0.0, 0.0, state, reason)
        return self.last

    def update(self, obs: TargetObservation | None, target_selected: bool,
               stale: bool, now: float) -> Guidance:
        dt = 0.05 if self._last_t is None else max(1e-3, min(0.5, now - self._last_t))
        self._last_t = now
        c = self.cfg

        if self._estop:
            return self._zero(ControllerState.EMERGENCY_STOP, "latched; press reset")
        if self._manual:
            return self._zero(ControllerState.MANUAL_OVERRIDE, "operator has control")
        if stale:
            return self._zero(ControllerState.STALE_FRAME, "no fresh camera frame")
        if not target_selected:
            self._lost_since = None
            return self._zero(ControllerState.IDLE, "no target selected")
        if obs is None or obs.confidence < c.min_confidence:
            if self._lost_since is None:
                self._lost_since = now
            lost_for = now - self._lost_since
            msg = f"lost {lost_for:.1f}s, holding"
            if lost_for > c.lost_alert_s:
                msg += " - recommend hover/failsafe"
            return self._zero(ControllerState.TARGET_LOST, msg)
        self._lost_since = None

        ex = obs.cx - 0.5
        eh = c.desired_height - obs.height
        yaw = _clamp(c.yaw_gain_dps * _deadband(ex, c.deadband_xy), c.max_yaw_dps)
        lat = (_clamp(c.lateral_gain * _deadband(ex, c.deadband_xy), c.max_lateral_mps)
               if c.use_lateral else 0.0)
        fwd = _clamp(c.forward_gain * _deadband(eh, c.deadband_h), c.max_forward_mps)

        py, pl, pf = self._prev
        yaw = _slew(py, yaw, c.max_yaw_accel, dt)
        lat = _slew(pl, lat, c.max_lin_accel, dt)
        fwd = _slew(pf, fwd, c.max_lin_accel, dt)
        self._prev = (yaw, lat, fwd)
        self.last = Guidance(yaw, lat, fwd, ControllerState.FOLLOWING,
                             f"ex={ex:+.2f} eh={eh:+.2f}")
        return self.last


# ---- output interfaces ---------------------------------------------------
class DroneInterface(ABC):
    @abstractmethod
    def send(self, g: Guidance, dt: float) -> None: ...

    def emergency_stop(self) -> None:
        self.send(Guidance(state=ControllerState.EMERGENCY_STOP), 0.0)

    def close(self) -> None:
        pass


class GuidanceOnlyInterface(DroneInterface):
    """Stores the latest suggestion; nothing is sent anywhere."""
    def __init__(self):
        self.last = Guidance()

    def send(self, g: Guidance, dt: float) -> None:
        self.last = g


class SimulatedDrone(DroneInterface):
    """Integrates guidance into a 2-D pose so the effect can be inspected."""
    def __init__(self):
        self.x = self.y = 0.0
        self.heading_deg = 0.0
        self.last = Guidance()

    def send(self, g: Guidance, dt: float) -> None:
        self.last = g
        self.heading_deg = (self.heading_deg + g.yaw_rate_dps * dt) % 360.0
        r = math.radians(self.heading_deg)
        self.x += (g.forward_mps * math.sin(r) + g.lateral_mps * math.cos(r)) * dt
        self.y += (g.forward_mps * math.cos(r) - g.lateral_mps * math.sin(r)) * dt


class MavlinkDroneInterface(DroneInterface):
    """Placeholder for a future flight-controller bridge (e.g. MAVLink/pymavlink).

    Intentionally NOT implemented. Before implementing, you need: hardware
    kill switch, geofence, speed/altitude limits, heartbeat/link-loss failsafe,
    SITL then hardware-in-the-loop validation, and a safety pilot.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Physical drone control is disabled in this prototype.")

    def send(self, g: Guidance, dt: float) -> None:  # pragma: no cover
        raise NotImplementedError
