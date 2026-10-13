"""All tunable parameters in one place."""
from __future__ import annotations

from dataclasses import dataclass, field

from dronetrack.drone_controller import ControllerConfig


@dataclass
class Config:
    # ---- capture ----
    source: int | str = 0            # webcam index, or path to a video file
    frame_width: int = 640           # frames are resized to this before anything else
    frame_height: int = 480
    queue_size: int = 2              # bounded; oldest frame is dropped when full
    stale_frame_s: float = 0.5       # frame older than this => STALE

    # ---- detection / ByteTrack ----
    model_path: str = "yolov8n.pt"   # downloaded automatically on first run
    imgsz: int = 320                 # YOLO inference size (smaller = faster)
    conf: float = 0.35
    infer_every_n: int = 1           # run YOLO every Nth frame (others reuse last result)
    tracker: str = "bytetrack.yaml"
    torch_threads: int = 4
    min_box_px: int = 20

    # ---- target selection / ReID recovery ----
    hist_update_every: int = 15      # refresh target appearance every N frames (when not occluded)
    reid_every_n: int = 3            # try recovery every N frames while target is missing
    reid_min_sim: float = 0.50       # minimum appearance similarity (0..1)
    reid_min_score: float = 0.60     # combined appearance + proximity score
    reid_search_frac: float = 0.50   # search radius as a fraction of frame diagonal
    lost_timeout_s: float = 3.0      # SEARCHING -> LOST (warning) after this

    # ---- movement analysis ----
    smooth_alpha: float = 0.35       # EMA factor (lower = smoother, more lag)
    analysis_window: int = 8         # samples used to measure displacement
    move_thresh: float = 0.02        # fraction of frame size moved over the window
    scale_thresh: float = 0.06       # relative box-height change over the window
    confirm_frames: int = 3          # label must persist this many updates to switch
    person_height_m: float = 1.7     # ASSUMED height, used only for rough m/s estimate
    edge_margin_px: int = 4          # box touching the frame edge => no m/s estimate

    # ---- ego-motion (camera/drone movement) ----
    ego_enabled: bool = True
    ego_width: int = 160
    ego_height: int = 120
    ego_min_response: float = 0.10   # phase-correlation reliability threshold
    cam_motion_px: float = 1.5       # mean px/frame of background shift => "camera moving"

    # ---- warnings / logging ----
    low_conf_warn: float = 0.40
    db_path: str = "data/movement.db"
    log_hz: float = 5.0

    controller: ControllerConfig = field(default_factory=ControllerConfig)
