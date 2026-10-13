"""Lightweight appearance descriptor for recovering the target after occlusion.

Uses an HSV color histogram of the torso and legs. It is cheap (no neural net)
and is computed only for the target (periodically) and for candidate tracks
while the target is missing.

To use a neural ReID model (e.g. OSNet) later, replace compute_appearance() and
appearance_similarity() - selection.py only depends on these two functions.

Limitation: color histograms fail for people in similar clothing or under
strong lighting changes.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_appearance(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    bw = x2 - x1
    margin = int(bw * 0.15)  # trim background at the sides of the box
    crop = frame[y1:y2, x1 + margin:x2 - margin]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hists = []
    for part in (hsv[8:32], hsv[32:60]):  # torso, legs (head skipped)
        hist = cv2.calcHist([part], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        hists.append(hist)
    return hists


def appearance_similarity(a, b) -> float:
    """0..1, higher = more similar."""
    if a is None or b is None:
        return 0.0
    sims = [max(0.0, cv2.compareHist(x, y, cv2.HISTCMP_CORREL)) for x, y in zip(a, b)]
    return float(np.mean(sims))


def blend(old, new, alpha: float = 0.2):
    """Exponential moving average of two descriptors."""
    if old is None:
        return new
    if new is None:
        return old
    return [(1 - alpha) * o + alpha * n for o, n in zip(old, new)]
