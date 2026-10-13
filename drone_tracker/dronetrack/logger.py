"""SQLite movement log. Samples are rate-limited; direction changes and
state events (selected, lost, recovered, ...) are always stored."""
from __future__ import annotations

import os
import sqlite3
import time
from collections import deque

SCHEMA = """
CREATE TABLE IF NOT EXISTS movement_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT, ts REAL, ts_iso TEXT, event TEXT, target_id INTEGER, status TEXT,
    cx REAL, cy REAL, box_h REAL,
    dir_image TEXT, dir_world TEXT,
    speed_norm REAL, speed_mps REAL,
    track_conf REAL, dir_conf REAL, camera_moving INTEGER
);
CREATE INDEX IF NOT EXISTS idx_session_ts ON movement_events(session, ts);
"""
FIELDS = ["target_id", "status", "cx", "cy", "box_h", "dir_image", "dir_world",
          "speed_norm", "speed_mps", "track_conf", "dir_conf", "camera_moving"]


class MovementLogger:
    def __init__(self, path: str, log_hz: float = 5.0):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.min_dt = 1.0 / max(0.1, log_hz)
        self._buf: list[tuple] = []
        self._last_sample = 0.0
        self._last_dir = None
        self._last_commit = time.time()
        self.recent: deque[str] = deque(maxlen=12)   # shown in the dashboard

    def log(self, event: str = "sample", **f) -> None:
        now = time.time()
        if event == "sample":
            changed = f.get("dir_image") != self._last_dir
            if not changed and now - self._last_sample < self.min_dt:
                return
            if changed and self._last_dir is not None:
                event = "change"
            self._last_sample, self._last_dir = now, f.get("dir_image")
        row = (self.session, now, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), event,
               *[f.get(k) for k in FIELDS])
        self._buf.append(row)
        if event != "sample":
            label = f.get("dir_image") if event == "change" else event.upper()
            self.recent.append(f"{time.strftime('%H:%M:%S')} id={f.get('target_id')} {label}")
        if now - self._last_commit > 1.0:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self.conn.executemany(
                "INSERT INTO movement_events(session,ts,ts_iso,event,target_id,status,cx,cy,box_h,"
                "dir_image,dir_world,speed_norm,speed_mps,track_conf,dir_conf,camera_moving) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", self._buf)
            self.conn.commit()
            self._buf.clear()
        self._last_commit = time.time()

    def close(self) -> None:
        self.flush()
        self.conn.close()
