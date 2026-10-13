"""Drone-based person tracking prototype (guidance-only / simulation).

Run:  python main.py                (webcam 0)
      python main.py --source clip.mp4
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from dronetrack.capture import FrameGrabber
from dronetrack.config import Config
from dronetrack.dashboard import Dashboard
from dronetrack.detection import PersonTracker
from dronetrack.drone_controller import FollowController, SimulatedDrone, TargetObservation
from dronetrack.logger import MovementLogger
from dronetrack.metrics import Metrics
from dronetrack.movement import EgoMotion, MovementAnalyzer
from dronetrack.selection import TargetSelector


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--model", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--infer-every", type=int, default=None, help="run YOLO every Nth frame")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--db", default=None)
    p.add_argument("--no-ego", action="store_true", help="disable camera-motion compensation")
    a = p.parse_args()
    cfg = Config()
    cfg.source = int(a.source) if a.source.isdigit() else a.source
    if a.model: cfg.model_path = a.model
    if a.imgsz: cfg.imgsz = a.imgsz
    if a.infer_every: cfg.infer_every_n = a.infer_every
    if a.width: cfg.frame_width = a.width
    if a.height: cfg.frame_height = a.height
    if a.threads: cfg.torch_threads = a.threads
    if a.db: cfg.db_path = a.db
    if a.no_ego: cfg.ego_enabled = False
    return cfg


def main() -> None:
    cfg = parse_args()
    grabber = FrameGrabber(cfg).start()
    detector = PersonTracker(cfg)
    selector = TargetSelector(cfg)
    ego = EgoMotion(cfg)
    analyzer = MovementAnalyzer(cfg)
    ctrl = FollowController(cfg.controller)
    drone = SimulatedDrone()          # guidance-only: no real hardware is ever commanded
    logger = MovementLogger(cfg.db_path, cfg.log_hz)
    metrics = Metrics()
    dash = Dashboard(cfg)

    frame = np.zeros((cfg.frame_height, cfg.frame_width, 3), np.uint8)
    tracks: list = []
    info: dict = {"status": "NONE"}
    last_res = None
    paused, id_buf = False, ""
    prev_t = time.perf_counter()
    print("Running. Click a person (or type an ID + Enter) to select a target. 'q' quits.")

    def on_selected(tr) -> None:
        analyzer.reset()
        logger.log("selected", target_id=tr.track_id, status="TRACKING")
        print(f"Target selected: ID {tr.track_id}")

    try:
        while True:
            now = time.perf_counter()
            dt, prev_t = now - prev_t, now

            if not paused:
                item = grabber.get(timeout=0.3)
                if item is None:
                    if grabber.eof:
                        break
                    g = ctrl.update(None, selector.target_id is not None, True, now)
                    drone.send(g, dt)
                    info.update(warning="NO CAMERA FRAME (stale)", ctrl_state=g.state.value,
                                guidance=g.reason)
                else:
                    frame, cap_ts = item
                    fh, fw = frame.shape[:2]
                    tracks, ran = detector.update(frame)
                    now = time.perf_counter()
                    target, event = selector.update(tracks, frame, now)
                    cam_cum, cam_moving = (ego.update(frame, [t.box for t in tracks])
                                           if cfg.ego_enabled else (np.zeros(2), False))

                    if event in ("recovered", "reacquired"):
                        analyzer.reset()
                    if target is None:
                        last_res = None
                    elif ran or last_res is None:
                        last_res = analyzer.update(target.box, target.conf, now, fw, fh,
                                                   cam_cum, cam_moving)
                    res = last_res if target is not None else None

                    obs = (TargetObservation(res.cx, res.cy, res.box_h, target.conf)
                           if (target is not None and res is not None) else None)
                    stale = (now - cap_ts) > cfg.stale_frame_s
                    g = ctrl.update(obs, selector.target_id is not None, stale, now)
                    drone.send(g, dt)

                    if event:
                        logger.log(event, target_id=selector.target_id, status=selector.state)
                    if res is not None and res.ready:
                        logger.log("sample", target_id=selector.target_id, status=selector.state,
                                   cx=res.cx, cy=res.cy, box_h=res.box_h,
                                   dir_image=res.direction_image, dir_world=res.direction_world,
                                   speed_norm=res.speed_norm, speed_mps=res.speed_mps,
                                   track_conf=target.conf, dir_conf=res.confidence,
                                   camera_moving=int(res.camera_moving))

                    metrics.frame(cap_ts, detector.last_infer_ms if ran else None)

                    warns = []
                    if ctrl.estop: warns.append("EMERGENCY STOP")
                    if stale: warns.append("STALE FRAMES")
                    if selector.state in ("SEARCHING", "LOST"):
                        warns.append("TARGET LOST - searching" if selector.state == "SEARCHING"
                                     else "TARGET LOST > %.0fs" % cfg.lost_timeout_s)
                    elif target is not None and target.conf < cfg.low_conf_warn:
                        warns.append("tracking confidence low")
                    if res is not None and res.camera_moving:
                        warns.append("camera moving: direction less reliable")
                    info.update(
                        status=selector.state, warning=warns[0] if warns else None,
                        dir_image=res.direction_image if res else "-",
                        dir_world=res.direction_world if res else "-",
                        dir_conf=res.confidence if res else 0.0,
                        track_conf=target.conf if target else 0.0,
                        speed_mps=res.speed_mps if res else None,
                        cam_moving=cam_moving, ctrl_state=g.state.value,
                        guidance=f"yaw {g.yaw_rate_dps:+.1f}d/s lat {g.lateral_mps:+.2f} fwd {g.forward_mps:+.2f}",
                        sim_x=drone.x, sim_y=drone.y, sim_hdg=drone.heading_deg,
                        fps=metrics.fps, cpu=metrics.cpu_pct, latency_ms=metrics.latency_ms,
                        infer_ms=metrics.infer_ms)
            info["paused"] = paused
            info["id_buf"] = id_buf

            # ---- operator input ----
            click = dash.pop_click()
            if click:
                tr = selector.select_at(click[0], click[1], tracks, frame)
                if tr:
                    on_selected(tr)

            key = dash.show(dash.render(frame, tracks, selector.target_id, info, logger.recent))
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("d"):
                selector.deselect(); analyzer.reset(); last_res = None
                logger.log("deselected")
            elif key == ord("m"):
                ctrl.set_manual(not ctrl.manual)
                logger.log("manual_on" if ctrl.manual else "manual_off")
            elif key == ord("e"):
                ctrl.trigger_estop(); drone.emergency_stop(); logger.log("estop")
            elif key == ord("r"):
                ctrl.reset_estop(); logger.log("estop_reset")
            elif ord("0") <= key <= ord("9"):
                id_buf += chr(key)
            elif key == 8:
                id_buf = id_buf[:-1]
            elif key in (13, 10) and id_buf:
                tr = selector.select_id(int(id_buf), tracks, frame)
                if tr: on_selected(tr)
                else: print(f"ID {id_buf} not visible")
                id_buf = ""
    finally:
        grabber.stop()
        logger.close()
        dash.close()
        print(metrics.summary())
        print(f"Frames dropped by capture queue: {grabber.dropped}")
        print(f"Log: {cfg.db_path} (session {logger.session})  ->  python tools/show_history.py")


if __name__ == "__main__":
    main()
