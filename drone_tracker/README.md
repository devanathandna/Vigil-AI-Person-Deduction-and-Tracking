# Drone-Based Person Tracking – CPU Prototype

Detects people with YOLO nano, tracks them with ByteTrack, lets the operator pick one target,
classifies its movement, logs to SQLite, and computes **suggested** drone-following corrections.
**Guidance-only / simulation. It never commands real hardware.**

## Setup (Windows)

1. Install Python 3.10–3.12 (tick "Add to PATH").
2. In the project folder: `setup_windows.bat` (creates `.venv`, installs CPU PyTorch + dependencies).
3. `.venv\Scripts\activate`
4. `python main.py` (first run downloads `yolov8n.pt`; internet needed once).

Options: `--source 1` (other webcam) · `--source clip.mp4` · `--imgsz 256` · `--infer-every 2` ·
`--width 480 --height 360` · `--threads 2` · `--no-ego`. All defaults live in `dronetrack/config.py`.

## Controls

| Input | Action |
|---|---|
| Click a box | Select / change target |
| Digits + Enter | Select by tracking ID |
| `d` | Deselect |
| `p` | Pause / resume |
| `m` | Toggle manual override (guidance forced to zero) |
| `e` / `r` | Emergency stop (latched) / reset |
| `q` or Esc | Quit (prints measured FPS / CPU / latency) |

## Architecture

```
FrameGrabber (thread, bounded queue, drops old frames)
   -> PersonTracker      YOLOv8n (person class) + ByteTrack IDs        detection.py
   -> TargetSelector     click/ID select, occlusion recovery           selection.py + reid.py
   -> EgoMotion          camera shift from background (people masked)  movement.py
   -> MovementAnalyzer   smoothing, thresholds, direction, speed       movement.py
   -> MovementLogger     SQLite                                        logger.py
   -> FollowController   target offset -> yaw/lateral/forward          drone_controller.py
   -> DroneInterface     SimulatedDrone / GuidanceOnly (Mavlink stub is disabled)
   -> Dashboard          OpenCV window                                 dashboard.py
```

`drone_controller.py` imports nothing from the vision code. It receives a `TargetObservation`
(normalized x, y, box height, confidence) plus `target_selected` / `stale` flags, so a flight
controller bridge can be added later without touching detection.

**Performance measures:** frames are resized to 640x480; YOLO runs at `imgsz=320`; queue size is 2;
`infer_every_n` can skip inference; ReID descriptors are computed only for the target (every 15 frames,
when not overlapped) and for new candidate tracks while the target is missing (every 3rd frame).

**Movement logic**
- Box center and height are smoothed (EMA), then displacement over a window of 8 samples is compared with
  thresholds (fraction of frame size). A new label must persist 3 updates before it is shown (debounce).
- *Left/right/up/down* come from center motion; *approaching/away* from change in box height.
- **Image vs scene direction:** global background shift is estimated by phase correlation and subtracted,
  giving a "scene direction". If the camera pans, image direction changes while scene direction should stay
  STATIONARY. A "camera moving" flag lowers confidence.
- Speed is reported only when scene motion is detected; m/s is a rough estimate that assumes a 1.7 m person
  and is withheld when the box touches the frame edge.

**Drone safety states:** IDLE, FOLLOWING, TARGET_LOST, STALE_FRAME, MANUAL_OVERRIDE, EMERGENCY_STOP.
Every non-FOLLOWING state outputs zero immediately; FOLLOWING output is deadbanded, clamped and slew-limited.

## Movement history

`python tools/show_history.py` (latest session) · `--all-events` includes periodic samples ·
`--session ID --limit 200`. Table `movement_events` has: timestamp, event, target ID, status, normalized
position, image direction, scene direction, speed, tracking confidence, direction confidence, camera-moving flag.

## Limitations (please read)

- **ReID is a color histogram, not a neural network.** It fails for similar clothing or lighting changes.
  Swap `reid.py` for OSNet or similar if needed.
- **ID swaps are not detected.** If ByteTrack gives the target's ID to another person during a crossing,
  the system follows the wrong person. Recovery only handles the case where the target gets a *new* ID.
- **Ego-motion is translation-only** and assumes the background dominates the frame. It does not capture
  forward/backward drone motion, roll, or strong parallax. Approach/away is therefore *apparent* only:
  a drone flying toward a stationary person will look like "approaching".
- Box height is a crude distance proxy (posture, partial visibility, and edge clipping change it).
- The webcam is not a drone camera: vibration, latency, and gimbal behavior are untested.
- Thresholds are starting points and need tuning for your camera and scene.
- Written without access to a webcam or Ultralytics in the development sandbox: unit tests for
  movement, controller and logger pass, but **the full camera pipeline has not been run by the author**.
  Treat the first run as part of testing, and report any errors.

## Testing procedure

Record numbers from your own machine; none are claimed here.

1. **Unit tests (no camera):** `python -m unittest discover -s tests -v`
2. **Stage 1 – detection:** run `python main.py`, no selection. Check boxes and IDs on 1–3 people.
   Close with `q` and note the printed FPS, latency, CPU, RAM.
3. **Tuning for speed:** if FPS is low, try `--imgsz 256`, `--infer-every 2`, `--width 480 --height 360`,
   `--threads 2`/`6`. Re-measure each time.
4. **Stage 2 – selection:** click a person, then walk someone across in front. Note how often the ID is kept,
   recovered (`RECOVERED` in history) or wrong.
5. **Stage 3 – direction:** walk left/right, toward/away, stand still. Compare labels and note false positives.
   Then hold the laptop and pan it slowly with the person standing still: image direction should change while
   scene direction stays STATIONARY. If the sign looks reversed on your setup, compare with `--no-ego`.
6. **Stage 4 – logging:** run `tools/show_history.py` and check it against what you did.
7. **Stage 5 – simulated guidance:** watch yaw/forward values and the simulated pose. Verify target right of
   center -> positive yaw; small box -> forward; cover the camera -> STALE_FRAME; hide the person -> TARGET_LOST;
   `m` and `e` -> zero output.
8. **Stage 6 – hardware-in-the-loop (not included):** ArduPilot/PX4 SITL first, then props-off bench with a
   hardware kill switch, then tethered/controlled flight with a safety pilot, geofence and speed limits.
   Implement a new `DroneInterface` for this; `MavlinkDroneInterface` is a deliberate stub.
