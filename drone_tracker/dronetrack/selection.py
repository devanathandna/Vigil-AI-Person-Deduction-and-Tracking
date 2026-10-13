"""Target selection and occlusion recovery.

States: NONE -> TRACKING -> SEARCHING -> LOST (still searching, warning shown).

ByteTrack keeps the same ID through short occlusions when it can. When it can't,
the target gets a *new* ID. Recovery therefore only considers tracks that first
appeared after the target was last seen (so an already-known bystander is never
"stolen"), scored by appearance similarity and proximity to the last position.

Known limitation: if ByteTrack swaps IDs while two people overlap, this module
cannot detect that the ID now belongs to the wrong person.
"""
from __future__ import annotations

import math

from dronetrack.detection import Track
from dronetrack.reid import appearance_similarity, blend, compute_appearance


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


class TargetSelector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.target_id: int | None = None
        self.state = "NONE"
        self.hist = None
        self.last_box = None
        self.last_seen_t = 0.0
        self.last_seen_frame = 0
        self._last_hist_frame = 0
        self._first_seen: dict[int, int] = {}
        self._frame = 0

    # ---- operator actions --------------------------------------------
    def select_at(self, x: float, y: float, tracks: list[Track], frame) -> Track | None:
        hits = [t for t in tracks if t.box[0] <= x <= t.box[2] and t.box[1] <= y <= t.box[3]]
        if not hits:
            return None
        pick = min(hits, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        self._set_target(pick, frame)
        return pick

    def select_id(self, tid: int, tracks: list[Track], frame) -> Track | None:
        for t in tracks:
            if t.track_id == tid:
                self._set_target(t, frame)
                return t
        return None

    def deselect(self) -> None:
        self.target_id = None
        self.state = "NONE"
        self.hist = None
        self.last_box = None

    def _set_target(self, t: Track, frame) -> None:
        import time
        self.target_id = t.track_id
        self.state = "TRACKING"
        self.last_box = t.box
        self.last_seen_t = time.perf_counter()
        self.last_seen_frame = self._frame
        self._last_hist_frame = self._frame
        self.hist = compute_appearance(frame, t.box)

    # ---- per-frame update --------------------------------------------
    def update(self, tracks: list[Track], frame, now: float):
        """Returns (target_track_or_None, event_or_None).
        Events: 'lost', 'reacquired' (same ID back), 'recovered' (new ID matched)."""
        c = self.cfg
        self._frame += 1
        for t in tracks:
            self._first_seen.setdefault(t.track_id, self._frame)
        if len(self._first_seen) > 1000:  # keep memory bounded
            live = {t.track_id for t in tracks}
            self._first_seen = {k: v for k, v in self._first_seen.items()
                                if k in live or v > self._frame - 600}

        if self.target_id is None:
            return None, None

        cur = next((t for t in tracks if t.track_id == self.target_id), None)
        if cur is not None:
            event = "reacquired" if self.state in ("SEARCHING", "LOST") else None
            self.state = "TRACKING"
            self.last_box, self.last_seen_t, self.last_seen_frame = cur.box, now, self._frame
            if self._frame - self._last_hist_frame >= c.hist_update_every:
                if not any(o.track_id != cur.track_id and _iou(o.box, cur.box) > 0.10
                           for o in tracks):  # skip when overlapped: would pollute template
                    self.hist = blend(self.hist, compute_appearance(frame, cur.box))
                    self._last_hist_frame = self._frame
            return cur, event

        event = None
        if self.state == "TRACKING":
            self.state = "SEARCHING"
            event = "lost"
        if now - self.last_seen_t > c.lost_timeout_s:
            self.state = "LOST"

        if self.hist is not None and self._frame % max(1, c.reid_every_n) == 0:
            cand = self._best_candidate(tracks, frame)
            if cand is not None:
                self.target_id = cand.track_id
                self.state = "TRACKING"
                self.last_box, self.last_seen_t, self.last_seen_frame = cand.box, now, self._frame
                return cand, "recovered"
        return None, event

    def _best_candidate(self, tracks: list[Track], frame) -> Track | None:
        c = self.cfg
        fh, fw = frame.shape[:2]
        diag = math.hypot(fw, fh)
        lx, ly = _center(self.last_box)
        best, best_score = None, 0.0
        for t in tracks:
            if self._first_seen.get(t.track_id, 0) < self.last_seen_frame:
                continue  # existed before the target vanished -> a different person
            sim = appearance_similarity(self.hist, compute_appearance(frame, t.box))
            if sim < c.reid_min_sim:
                continue
            cx, cy = _center(t.box)
            prox = max(0.0, 1.0 - math.hypot(cx - lx, cy - ly) / (c.reid_search_frac * diag))
            score = 0.7 * sim + 0.3 * prox
            if score > best_score:
                best, best_score = t, score
        return best if best is not None and best_score >= c.reid_min_score else None
