"""Movement analysis for the selected target.

Two things are computed:
  * IMAGE direction  : how the box moves in the camera image (what the follow
                       controller reacts to).
  * SCENE direction  : image motion minus estimated camera (ego) motion, i.e.
                       an approximation of how the person moves relative to the
                       background. If the camera pans, the image direction
                       changes but the scene direction should stay STATIONARY.

Ego-motion is a global 2-D translation estimated with phase correlation on a
small grayscale frame, with person boxes masked out. It cannot detect forward
camera motion, rotation about the optical axis, or strong parallax.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


class EgoMotion:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = (cfg.ego_width, cfg.ego_height)
        self.win = cv2.createHanningWindow(self.size, cv2.CV_32F)
        self.prev = None
        self.cum = np.zeros(2, dtype=np.float64)   # cumulative background shift, full-frame px
        self.recent = deque(maxlen=cfg.analysis_window)

    def update(self, frame, boxes):
        """Returns (cumulative_shift_xy, camera_moving)."""
        w, h = self.size
        fh, fw = frame.shape[:2]
        g = cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = float(g.mean())
        for x1, y1, x2, y2 in boxes:  # hide people so they don't bias the estimate
            g[max(0, int(y1 * h / fh)):int(y2 * h / fh) + 1,
              max(0, int(x1 * w / fw)):int(x2 * w / fw) + 1] = mean
        if self.prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(self.prev, g, self.win)
            if resp >= self.cfg.ego_min_response:
                sx, sy = dx * fw / w, dy * fh / h
                self.cum += (sx, sy)
                self.recent.append(math.hypot(sx, sy))
            else:
                self.recent.append(0.0)
        self.prev = g
        moving = bool(self.recent) and float(np.mean(self.recent)) > self.cfg.cam_motion_px
        return self.cum.copy(), moving


class _Debounce:
    """A new label must persist for n consecutive updates before it is adopted."""
    def __init__(self, n: int):
        self.n, self.cur, self.pend, self.count = max(1, n), "STATIONARY", None, 0

    def push(self, v: str) -> str:
        if v == self.cur:
            self.pend, self.count = None, 0
        elif v == self.pend:
            self.count += 1
        else:
            self.pend, self.count = v, 1
        if self.pend is not None and self.count >= self.n:
            self.cur, self.pend, self.count = self.pend, None, 0
        return self.cur


@dataclass
class MovementResult:
    ready: bool
    direction_image: str
    direction_world: str
    confidence: float
    speed_norm: float | None   # frame-widths per second, scene-relative
    speed_mps: float | None    # rough estimate, assumes cfg.person_height_m
    camera_moving: bool
    cx: float                  # smoothed center, normalized 0..1
    cy: float
    box_h: float               # smoothed box height / frame height


class MovementAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._s = None
        self._hist: deque = deque(maxlen=self.cfg.analysis_window)
        self._db_img = _Debounce(self.cfg.confirm_frames)
        self._db_world = _Debounce(self.cfg.confirm_frames)

    def _label(self, dx: float, dy: float, ds: float):
        c = self.cfg
        parts = []
        if dx < -c.move_thresh:
            parts.append("LEFT")
        elif dx > c.move_thresh:
            parts.append("RIGHT")
        if dy < -c.move_thresh:
            parts.append("UP")
        elif dy > c.move_thresh:
            parts.append("DOWN")
        if ds > c.scale_thresh:
            parts.append("APPROACHING")
        elif ds < -c.scale_thresh:
            parts.append("AWAY")
        ratio = max(abs(dx) / c.move_thresh, abs(dy) / c.move_thresh, abs(ds) / c.scale_thresh)
        return ("+".join(parts) if parts else "STATIONARY"), ratio

    def update(self, box, det_conf: float, t: float, fw: int, fh: int,
               ego_cum, cam_moving: bool) -> MovementResult:
        c = self.cfg
        raw = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[3] - box[1]], float)
        if self._s is None:
            self._s = raw.copy()
        else:
            self._s += c.smooth_alpha * (raw - self._s)
        sx, sy, sh = self._s
        self._hist.append((t, sx, sy, sh, sx - ego_cum[0], sy - ego_cum[1]))

        n = len(self._hist)
        if n < max(3, c.analysis_window // 2):
            return MovementResult(False, "WARMING_UP", "WARMING_UP", 0.0, None, None,
                                  cam_moving, sx / fw, sy / fh, sh / fh)

        a, b = self._hist[0], self._hist[-1]
        dur = max(1e-3, b[0] - a[0])
        ds = math.log(max(b[3], 1.0) / max(a[3], 1.0))
        img_label, ratio = self._label((b[1] - a[1]) / fw, (b[2] - a[2]) / fh, ds)
        world_label, _ = self._label((b[4] - a[4]) / fw, (b[5] - a[5]) / fh, ds)
        img_label = self._db_img.push(img_label)
        world_label = self._db_world.push(world_label)

        base = (1.0 - min(1.0, ratio)) if img_label == "STATIONARY" else min(1.0, ratio / 2.0)
        conf = float(det_conf * base * (0.6 if cam_moving else 1.0))

        speed_norm = speed_mps = 0.0
        if world_label != "STATIONARY":
            px = math.hypot(b[4] - a[4], b[5] - a[5]) / dur
            speed_norm = px / fw
            touching = (box[0] < c.edge_margin_px or box[1] < c.edge_margin_px or
                        box[2] > fw - c.edge_margin_px or box[3] > fh - c.edge_margin_px)
            speed_mps = None if touching else px * c.person_height_m / max(sh, 1.0)
        return MovementResult(True, img_label, world_label, conf, speed_norm, speed_mps,
                              cam_moving, sx / fw, sy / fh, sh / fh)
