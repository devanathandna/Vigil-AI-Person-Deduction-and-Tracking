"""OpenCV dashboard: video with boxes on the left, status panel on the right."""
from __future__ import annotations

import time

import cv2
import numpy as np

PANEL_W = 330
WIN = "Drone Person Tracking"
ORANGE, GREEN, RED, WHITE, GRAY, YELLOW = (0, 140, 255), (0, 200, 0), (0, 0, 255), (240, 240, 240), (150, 150, 150), (0, 220, 255)

HELP = "click=select  0-9+Enter=select ID  d=deselect  p=pause  m=manual  e=E-STOP  r=reset  q=quit"


class Dashboard:
    def __init__(self, cfg):
        self.w, self.h = cfg.frame_width, cfg.frame_height
        self._click = None
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and x < self.w:
            self._click = (x, y)

    def pop_click(self):
        c, self._click = self._click, None
        return c

    def render(self, frame, tracks, target_id, info: dict, history) -> np.ndarray:
        img = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.box)
            is_t = t.track_id == target_id
            col = ORANGE if is_t else GREEN
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if is_t else 1)
            label = f"ID {t.track_id}" + (" TARGET" if is_t else "")
            cv2.putText(img, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2 if is_t else 1)
            if is_t:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.drawMarker(img, (cx, cy), ORANGE, cv2.MARKER_CROSS, 14, 2)
        # frame-center crosshair (what the controller tries to align with)
        cv2.drawMarker(img, (self.w // 2, self.h // 2), GRAY, cv2.MARKER_CROSS, 20, 1)

        warn = info.get("warning")
        if warn:
            blink = int(time.time() * 2) % 2 == 0
            cv2.rectangle(img, (0, 0), (self.w, 30), RED if blink else (0, 0, 140), -1)
            cv2.putText(img, "WARNING: " + warn, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
        if info.get("paused"):
            cv2.putText(img, "PAUSED", (self.w // 2 - 60, self.h // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, YELLOW, 3)
        cv2.putText(img, HELP, (6, self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.37, WHITE, 1)

        panel = np.full((self.h, PANEL_W, 3), 30, np.uint8)
        y = [18]

        def put(text, col=WHITE, scale=0.46):
            cv2.putText(panel, text, (8, y[0]), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1)
            y[0] += 18

        status = info.get("status", "NONE")
        put(f"TARGET ID : {target_id if target_id is not None else '-'}"
            + (f"   (typing: {info['id_buf']})" if info.get("id_buf") else ""), ORANGE)
        put(f"Status    : {status}", RED if status in ("SEARCHING", "LOST") else GREEN)
        put(f"Image dir : {info.get('dir_image', '-')}")
        put(f"Scene dir : {info.get('dir_world', '-')}")
        put(f"Dir conf  : {info.get('dir_conf', 0.0):.2f}   Track conf: {info.get('track_conf', 0.0):.2f}")
        sm = info.get("speed_mps")
        put(f"Speed     : {'%.2f m/s (rough)' % sm if sm is not None else 'n/a'}")
        put("Camera moving" if info.get("cam_moving") else "Camera steady",
            YELLOW if info.get("cam_moving") else GRAY)
        y[0] += 4
        put(f"Controller: {info.get('ctrl_state', '-')}", YELLOW)
        put(f"  {info.get('guidance', '')}", GRAY, 0.42)
        put(f"  sim pose x={info.get('sim_x', 0):+.1f} y={info.get('sim_y', 0):+.1f} hdg={info.get('sim_hdg', 0):.0f}", GRAY, 0.42)
        y[0] += 4
        put(f"FPS {info.get('fps', 0):.1f}  CPU {info.get('cpu', 0):.0f}%", GRAY, 0.42)
        put(f"Latency {info.get('latency_ms', 0):.0f} ms  Infer {info.get('infer_ms', 0):.0f} ms", GRAY, 0.42)
        y[0] += 4
        put("History:", WHITE)
        room = max(0, (self.h - y[0]) // 15 - 1)
        for line in list(history)[-room:] if room else []:
            cv2.putText(panel, line, (8, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1)
            y[0] += 15
        return np.hstack([img, panel])

    @staticmethod
    def show(img) -> int:
        cv2.imshow(WIN, img)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()
