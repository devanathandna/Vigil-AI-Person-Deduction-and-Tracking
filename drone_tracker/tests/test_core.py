"""Camera-free sanity tests: python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dronetrack.config import Config
from dronetrack.drone_controller import FollowController, TargetObservation
from dronetrack.logger import MovementLogger
from dronetrack.movement import MovementAnalyzer


def run(box_fn, ego_fn=lambda i: np.zeros(2), n=30):
    an = MovementAnalyzer(Config())
    res = None
    for i in range(n):
        res = an.update(box_fn(i), 0.9, i * 0.1, 640, 480, ego_fn(i), False)
    return res


class TestMovement(unittest.TestCase):
    def test_stationary(self):
        r = run(lambda i: (100, 100, 200, 300))
        self.assertEqual(r.direction_image, "STATIONARY")

    def test_right(self):
        r = run(lambda i: (100 + 5 * i, 100, 200 + 5 * i, 300))
        self.assertIn("RIGHT", r.direction_image)

    def test_approaching(self):
        r = run(lambda i: (300, 100, 400, 100 + 200 * 1.02 ** i))
        self.assertIn("APPROACHING", r.direction_image)

    def test_camera_pan_is_not_world_motion(self):
        # box moves right 5 px/frame in the image, but the background also shifted 5 px/frame
        r = run(lambda i: (100 + 5 * i, 100, 200 + 5 * i, 300),
                ego_fn=lambda i: np.array([5.0 * i, 0.0]))
        self.assertIn("RIGHT", r.direction_image)
        self.assertEqual(r.direction_world, "STATIONARY")


class TestController(unittest.TestCase):
    def test_yaw_right_when_target_right(self):
        c = FollowController()
        g = None
        for i in range(10):
            g = c.update(TargetObservation(0.8, 0.5, 0.5, 0.9), True, False, i * 0.1)
        self.assertGreater(g.yaw_rate_dps, 0)
        self.assertAlmostEqual(g.forward_mps, 0.0)

    def test_forward_when_target_small(self):
        c = FollowController()
        for i in range(20):
            g = c.update(TargetObservation(0.5, 0.5, 0.2, 0.9), True, False, i * 0.1)
        self.assertGreater(g.forward_mps, 0)

    def test_safety_states_output_zero(self):
        c = FollowController()
        ok = TargetObservation(0.8, 0.5, 0.2, 0.9)
        c.update(ok, True, False, 0.0)
        self.assertEqual(c.update(None, True, False, 0.1).forward_mps, 0.0)   # lost
        self.assertEqual(c.update(ok, True, True, 0.2).yaw_rate_dps, 0.0)      # stale
        c.set_manual(True)
        self.assertEqual(c.update(ok, True, False, 0.3).state.value, "MANUAL_OVERRIDE")
        c.set_manual(False)
        c.trigger_estop()
        self.assertEqual(c.update(ok, True, False, 0.4).state.value, "EMERGENCY_STOP")
        c.reset_estop()
        self.assertEqual(c.update(ok, True, False, 0.5).state.value, "FOLLOWING")


class TestLogger(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            lg = MovementLogger(os.path.join(d, "t.db"))
            lg.log("selected", target_id=3, status="TRACKING")
            lg.log("sample", target_id=3, status="TRACKING", dir_image="LEFT")
            lg.flush()
            n = lg.conn.execute("SELECT COUNT(*) FROM movement_events").fetchone()[0]
            lg.close()
            self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
