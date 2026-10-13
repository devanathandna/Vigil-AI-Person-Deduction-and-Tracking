"""Threaded frame grabber with a small bounded queue.

For a live camera, old frames are dropped when the consumer is slow so latency
stays low. For a video file, the grabber blocks instead so every frame is used.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import cv2


class FrameGrabber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_file = isinstance(cfg.source, str)
        self.q: queue.Queue = queue.Queue(maxsize=cfg.queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.eof = False
        self.dropped = 0
        self.cap = None

    def start(self) -> "FrameGrabber":
        src = self.cfg.source
        if self.is_file:
            self.cap = cv2.VideoCapture(src)
        else:
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(src), backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        self._thread.start()
        return self

    def _run(self) -> None:
        size = (self.cfg.frame_width, self.cfg.frame_height)
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.eof = True
                    return
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            item = (frame, time.perf_counter())
            if self.is_file:
                while not self._stop.is_set():
                    try:
                        self.q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            else:
                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    try:
                        self.q.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
                    try:
                        self.q.put_nowait(item)
                    except queue.Full:
                        pass

    def get(self, timeout: float = 0.3):
        """Return (frame, capture_timestamp) or None on timeout."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
