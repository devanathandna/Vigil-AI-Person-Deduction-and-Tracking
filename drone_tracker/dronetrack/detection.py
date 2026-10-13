"""Person detection (YOLO nano) + ByteTrack IDs, using Ultralytics' built-in tracker."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from ultralytics import YOLO


@dataclass
class Track:
    track_id: int
    box: tuple          # (x1, y1, x2, y2) in processed-frame pixels
    conf: float


class PersonTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.set_num_threads(cfg.torch_threads)
        self.model = YOLO(cfg.model_path)
        self._n = 0
        self._last: list[Track] = []
        self.last_infer_ms = 0.0

    def update(self, frame) -> tuple[list[Track], bool]:
        """Returns (tracks, ran_inference). Skipped frames reuse the last result."""
        c = self.cfg
        self._n += 1
        if (self._n - 1) % max(1, c.infer_every_n) != 0:
            return self._last, False

        t0 = time.perf_counter()
        with torch.inference_mode():
            res = self.model.track(
                frame, persist=True, classes=[0], conf=c.conf, imgsz=c.imgsz,
                tracker=c.tracker, device="cpu", verbose=False,
            )[0]
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        tracks: list[Track] = []
        b = res.boxes
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.int().cpu().numpy()
            confs = b.conf.cpu().numpy()
            for box, tid, cf in zip(xyxy, ids, confs):
                if (box[2] - box[0]) >= c.min_box_px and (box[3] - box[1]) >= c.min_box_px:
                    tracks.append(Track(int(tid), tuple(float(v) for v in box), float(cf)))
        self._last = tracks
        return tracks, True
