"""Measures real FPS, process CPU usage, and latency. Nothing is estimated."""
from __future__ import annotations

import time

import numpy as np
import psutil


class Metrics:
    def __init__(self):
        self.proc = psutil.Process()
        self.ncpu = psutil.cpu_count() or 1
        self.proc.cpu_percent(None)
        self._t_last = None
        self._cpu_t = time.perf_counter()
        self.fps = 0.0
        self.cpu_pct = 0.0
        self.latency_ms = 0.0
        self.infer_ms = 0.0
        self._fps_hist: list[float] = []
        self._lat_hist: list[float] = []
        self._inf_hist: list[float] = []
        self._cpu_hist: list[float] = []

    def frame(self, capture_ts: float, infer_ms: float | None) -> None:
        now = time.perf_counter()
        if self._t_last is not None:
            inst = 1.0 / max(1e-6, now - self._t_last)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            self._fps_hist.append(inst)
        self._t_last = now
        self.latency_ms = (now - capture_ts) * 1000.0   # capture -> processed
        self._lat_hist.append(self.latency_ms)
        if infer_ms:
            self.infer_ms = infer_ms
            self._inf_hist.append(infer_ms)
        if now - self._cpu_t >= 1.0:   # whole-process CPU, normalized to 0..100% of all cores
            self.cpu_pct = self.proc.cpu_percent(None) / self.ncpu
            self._cpu_hist.append(self.cpu_pct)
            self._cpu_t = now

    def summary(self) -> str:
        def stat(v, unit):
            if not v:
                return "n/a"
            return (f"mean {np.mean(v):.1f}{unit}  p95 {np.percentile(v, 95):.1f}{unit}  "
                    f"max {np.max(v):.1f}{unit}")
        rss = self.proc.memory_info().rss / 1e6
        return ("--- measured performance (this run) ---\n"
                f"frames processed : {len(self._lat_hist)}\n"
                f"FPS              : mean {np.mean(self._fps_hist) if self._fps_hist else 0:.1f}\n"
                f"capture->processed latency : {stat(self._lat_hist, ' ms')}\n"
                f"YOLO+track inference       : {stat(self._inf_hist, ' ms')}\n"
                f"process CPU (all cores)    : {stat(self._cpu_hist, ' %')}\n"
                f"process RAM (RSS)          : {rss:.0f} MB")
