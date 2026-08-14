import os
import time
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Lock

# Load Torch before Qt/OpenCV. On Windows, loading Qt first can make Torch's
# c10.dll fail with WinError 1114 in entry points that import this module
# directly (the main application already does this in main.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
except ImportError:
    torch = None

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from vision.camera_config import load_camera_settings
from vision.distance_config import DISTANCE_SETTINGS_FILE, load_distance_thresholds
from vision.frame_utils import (
    letterbox_with_meta,
    unletterbox_box,
    unletterbox_points,
)
from vision.model_config import (
    DEFAULT_MODEL_CONF,
    DEFAULT_MODEL_IOU,
    DEFAULT_MODEL_RELATIVE,
    load_model_thresholds,
    normalize_model_thresholds,
)
from vision.object_height_config import (
    DEFAULT_OBJECT_HEIGHTS,
    load_object_heights,
)

# Resizing/drawing is lightweight here. Let Torch own the CPU worker pool instead
# of allowing OpenCV and Torch to oversubscribe the same cores.
cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / DEFAULT_MODEL_RELATIVE
ZONE_FILE = PROJECT_ROOT / "zones" / "active.txt"
FILE_CHECK_INTERVAL_S = 0.75

# Approximate real-world object heights in metres. Keep this public alias for
# compatibility with code that imports the calibration labels; the running
# detector uses the user-editable values in ``self._object_heights``.
REAL_HEIGHT = DEFAULT_OBJECT_HEIGHTS.copy()

CLASS_ALIASES = {
    "motorbike": "motorcycle",
    "bike": "motorcycle",
}

# Practical confidence thresholds for real webcam/video use.
# Higher values reduce false positives. Lower values detect more objects but may be noisy.
conf = DEFAULT_MODEL_CONF
CLASS_CONF = {
    "person": conf,
    "car": conf,
    "truck": conf,
    "motorcycle": conf,
}

# Keep the camera frame and the inference input square at exactly 640x640 so
# the inference coordinate system always matches the zone editor.
MODEL_IMGSZ = 640
MODEL_IOU = DEFAULT_MODEL_IOU
# Road scenes rarely have more than a handful of relevant objects in frame;
# capping detections lower trims NMS/post-processing work.
MODEL_MAX_DET = 20
# CPU inference is intentionally capped. The UI interpolates the last tracked
# boxes between detections, so leaving a short gap after inference is smoother
# overall than allowing inference to consume all CPU time.
CPU_INFERENCE_INTERVAL_S = 0.05

MIN_BOX_HEIGHT = 14
SMOOTHING_ALPHA = 0.35
# Lowered from 0.30: combined with velocity-based prediction below, a track
# only needs partial overlap with a fast object's *predicted* box, not its
# raw last-seen box, so this can be a little more forgiving without pairing
# up unrelated detections.
TRACK_IOU_THRESHOLD = 0.22
DANGER_HYSTERESIS_M = 0.60
WARNING_HYSTERESIS_M = 1.00

# How responsively the estimated box velocity reacts to the newest motion
# vs. the previous estimate. Higher = tracks sudden speed changes faster but
# is noisier frame-to-frame.
VELOCITY_SMOOTHING_ALPHA = 0.5
# Cap how far ahead a box position is ever extrapolated (both for matching a
# fast object to its previous track, and for drawing between inference
# results). Beyond this the track is treated as stale and shown/matched at
# its last known position instead of guessing further.
MAX_PREDICTION_S = 0.4
# Guard against a very small detector time delta producing an unrealistic
# pixel velocity. Such a prediction can send a box far outside the frame and
# make OpenCV spend excessive time clipping a draw operation.
MAX_TRACK_SPEED_PX_S = 2500.0

STATUS_STYLE = {
    "DANGER": {"color": (0, 0, 255), "html": "#ef4444", "thai": "อันตราย"},
    "WARNING": {"color": (0, 165, 255), "html": "#f59e0b", "thai": "ระวัง"},
    "SAFE": {"color": (0, 255, 0), "html": "#22c55e", "thai": "ปลอดภัย"},
}

LABEL_THAI = {
    "car": "รถยนต์",
    "truck": "รถบรรทุก",
    "motorcycle": "รถจักรยานยนต์",
    "person": "คน",
}


DEVICE_REQUEST = os.environ.get("THIRD_EYE_DEVICE", "auto").strip().lower()


def _select_device():
    """Prefer CUDA when the installed Torch build can actually use it.

    ``THIRD_EYE_DEVICE=cpu`` is useful for troubleshooting, while ``auto``
    keeps the normal installation portable.  A request for GPU still falls
    back safely when this machine has CPU-only Torch or no CUDA driver.
    """
    cuda_available = torch is not None and torch.cuda.is_available()
    if DEVICE_REQUEST in {"cpu", "none"}:
        return "cpu"
    if cuda_available:
        return "cuda"
    return "cpu"


def _normalize_label(name):
    label = str(name).strip().lower()
    return CLASS_ALIASES.get(label, label)


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


class YoloThread(QThread):
    """Runs inference only for the most recent camera/video frame."""

    result_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    model_ready_signal = pyqtSignal()

    def __init__(self, model_path=None):
        super().__init__()

        self.model_path = Path(model_path) if model_path else MODEL_FILE
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Heavy model loading is deferred to run() so the Qt UI remains
        # responsive while the model/device is initialized.
        self.device = None
        self.model = None
        self.class_ids = []
        self.model_ready = False
        model_thresholds = load_model_thresholds()
        self.conf = model_thresholds["conf"]
        self.iou = model_thresholds["iou"]

        self._stop_requested = Event()
        self._frame_ready = Event()
        self._lock = Lock()
        self._frame = None
        self._frame_uses_zone = True
        self._frame_id = 0
        self._processed_frame_id = 0
        self._source_generation = 0
        # Live capture is pulled directly by this worker. Keeping a single
        # latest-frame source prevents the GUI timer from becoming the
        # bottleneck that feeds inference.
        self._frame_source = None
        self._source_frame_number = -1
        self._frame_display_transform = None

        self.zone = None
        self.zone_mtime = 0
        self._zone_last_check = 0.0
        self.last_detections = []
        self._track_memory = []
        self.inference_fps = 0.0
        self._inference_count = 0
        self._inference_ms = 0.0
        self._inference_ms_avg = 0.0
        self._source_frames_skipped = 0
        self._thresholds = load_distance_thresholds()
        self._focal_length = load_camera_settings()["focal_length"]
        self._object_heights = load_object_heights()
        self._thresholds_mtime = 0
        self._thresholds_last_check = 0.0

    def _initialize_model(self):
        self.device = _select_device()
        if torch is not None:
            torch.set_grad_enabled(False)
            if self.device != "cuda":
                cpu_threads = max(1, min(4, (os.cpu_count() or 2) - 1))
                torch.set_num_threads(cpu_threads)
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
                if hasattr(torch.backends, "mkldnn"):
                    torch.backends.mkldnn.enabled = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        try:
            self.model.fuse()
        except (AttributeError, RuntimeError):
            pass
        class_names = (
            self.model.names.items()
            if isinstance(self.model.names, dict)
            else enumerate(self.model.names)
        )
        self.class_ids = [
            class_id for class_id, name in class_names
            if _normalize_label(name) in REAL_HEIGHT
        ]
        if not self.class_ids:
            supported = ", ".join(sorted(REAL_HEIGHT))
            raise ValueError(
                "โมเดลไม่มี class ที่ระบบรองรับ "
                f"({supported})"
            )
        self._warm_up()
        self.model_ready = True
        self.model_ready_signal.emit()

    def _warm_up(self):
        """Build the same inference path used by real frames."""
        dummy_frame = np.zeros(
            (MODEL_IMGSZ, MODEL_IMGSZ, 3), dtype=np.uint8
        )
        for _ in range(2):
            self._predict(dummy_frame)

    def _predict(self, frame):
        """Run inference with one shared set of production parameters."""
        inference_context = (
            torch.inference_mode() if torch is not None else nullcontext()
        )
        with inference_context:
            return self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=MODEL_IMGSZ,
                max_det=MODEL_MAX_DET,
                classes=self.class_ids or None,
                device=self.device,
                augment=False,
                verbose=False,
            )

    def update_frame(self, frame, use_zone=True):
        """Store the newest frame and discard older unprocessed frames."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return False
        with self._lock:
            if self._stop_requested.is_set():
                return False
            if self._frame_source is not None:
                # Live sources are pulled by run() in this worker. Rejecting
                # GUI-pushed copies keeps one source of truth for frame IDs.
                return False
            # CaptureThread publishes a frame reference and never mutates it
            # after publication. Keep the reference here; _detect() creates
            # its own 640x640 letterboxed buffer. Copying a 1080p frame on
            # every UI tick only consumed memory bandwidth and increased lag.
            self._frame = frame
            self._frame_uses_zone = bool(use_zone)
            self._frame_id += 1
        self._frame_ready.set()
        return True

    def set_frame_source(self, source):
        """Attach a thread-safe latest-frame provider for live sources.

        ``source`` may return ``(frame, frame_number)`` like
        ``CaptureThread.get_latest`` or ``(frame, frame_number, transform)``
        like ``CaptureThread.get_latest_model``. The provider is called only
        by this worker, while the GUI can independently read the same latest
        display frame.
        """
        with self._lock:
            self._frame_source = source
            self._source_frame_number = -1
            self._source_generation += 1
            self._frame = None
            self._frame_display_transform = None
            self._processed_frame_id = self._frame_id
            self.last_detections = []
            self._track_memory = []
        self._frame_ready.set()

    def clear_frame_source(self):
        """Detach a live provider before its capture thread is stopped."""
        with self._lock:
            self._frame_source = None
            self._source_frame_number = -1
            self._source_generation += 1
            self._frame = None
            self._frame_display_transform = None
            self._processed_frame_id = self._frame_id
            self.last_detections = []
            self._track_memory = []
        self._frame_ready.set()

    @property
    def frame_source_attached(self):
        with self._lock:
            return self._frame_source is not None

    def _pull_source_frame(self):
        """Pull one new frame from the attached provider, if available."""
        with self._lock:
            source = self._frame_source
        if source is None:
            return False
        try:
            source_result = source()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        if not isinstance(source_result, (tuple, list)):
            return False
        if len(source_result) == 2:
            frame, frame_number = source_result
            display_transform = None
        elif len(source_result) == 3:
            frame, frame_number, display_transform = source_result
        else:
            return False
        try:
            frame_number = int(frame_number)
        except (TypeError, ValueError):
            return False
        if frame is None or getattr(frame, "size", 0) == 0 or frame_number < 0:
            return False

        with self._lock:
            if source is not self._frame_source:
                return False
            if frame_number == self._source_frame_number:
                return False
            if self._source_frame_number >= 0 and frame_number > self._source_frame_number + 1:
                self._source_frames_skipped += (
                    frame_number - self._source_frame_number - 1
                )
            self._source_frame_number = frame_number
            self._frame = frame
            self._frame_display_transform = display_transform
            self._frame_uses_zone = True
            self._frame_id += 1
        return True

    def clear_frame(self):
        """Pause inference when no camera/video source is active."""
        with self._lock:
            self._source_generation += 1
            self._frame = None
            self._frame_display_transform = None
            self._source_frame_number = -1
            self._processed_frame_id = self._frame_id
            self.last_detections = []
            self._track_memory = []
        self._frame_ready.set()

    def load_zone(self, shape):
        if not ZONE_FILE.is_file():
            return None

        height, width = shape[:2]
        points = []
        with ZONE_FILE.open(encoding="utf-8") as file:
            for line in file:
                try:
                    x_ratio, y_ratio = map(float, line.strip().split(","))
                except ValueError:
                    continue
                if 0 <= x_ratio <= 1 and 0 <= y_ratio <= 1:
                    points.append((int(x_ratio * width), int(y_ratio * height)))

        return np.array(points, dtype=np.int32) if len(points) >= 3 else None

    def get_zone(self, source_shape=None):
        with self._lock:
            zone = None if self.zone is None else self.zone.copy()
        if zone is not None and source_shape is not None:
            return unletterbox_points(zone, source_shape, (MODEL_IMGSZ, MODEL_IMGSZ))
        return zone

    def get_detections(self, predict=True):
        """Return the latest detections.

        With `predict=True` (the default, used for live drawing) each box is
        advanced along its estimated velocity to the current moment, so a
        fast-moving object's box keeps pace with the video between inference
        results instead of freezing at its last detected position. Batch/
        CSV export code should pass `predict=False` to keep the exact
        detected box.
        """
        if not self.model_ready:
            return []
        with self._lock:
            detections = [detection.copy() for detection in self.last_detections]
        if not predict:
            return detections

        now = time.monotonic()
        for detection in detections:
            detection["box"] = self._predict_box(detection, now)
        return detections

    @staticmethod
    def _predict_box(track, now):
        timestamp = track.get("timestamp")
        vx, vy = track.get("velocity", (0.0, 0.0))
        if timestamp is None or (vx == 0.0 and vy == 0.0):
            return track["box"]

        elapsed = min(max(now - timestamp, 0.0), MAX_PREDICTION_S)
        if elapsed <= 0:
            return track["box"]

        x1, y1, x2, y2 = track["box"]
        vx = max(-MAX_TRACK_SPEED_PX_S, min(MAX_TRACK_SPEED_PX_S, float(vx)))
        vy = max(-MAX_TRACK_SPEED_PX_S, min(MAX_TRACK_SPEED_PX_S, float(vy)))
        shift_x = vx * elapsed
        shift_y = vy * elapsed
        return (
            int(round(x1 + shift_x)),
            int(round(y1 + shift_y)),
            int(round(x2 + shift_x)),
            int(round(y2 + shift_y)),
        )

    def run(self):
        try:
            self._initialize_model()
        except Exception as error:
            self.error_ready.emit(f"YOLO model load error: {error}")
            return
        last_inference_at = 0.0
        while not self._stop_requested.is_set():
            if self.frame_source_attached:
                # Polling one integer and one reference is cheap, and avoids
                # queuing every camera frame through Qt. The next iteration
                # always asks for the newest frame after inference completes.
                if not self._pull_source_frame():
                    if self._stop_requested.wait(0.01):
                        break
                    continue
            else:
                self._frame_ready.wait(timeout=0.1)

            if self.device == "cpu" and last_inference_at:
                remaining = CPU_INFERENCE_INTERVAL_S - (
                    time.monotonic() - last_inference_at
                )
                if remaining > 0:
                    if self._stop_requested.wait(remaining):
                        break
                    # Pick the newest frame after the throttle wait instead of
                    # processing a stale snapshot.
                    continue

            with self._lock:
                if self._frame is None or self._frame_id == self._processed_frame_id:
                    self._frame_ready.clear()
                    continue
                # update_frame already owns a private copy; replacing self._frame
                # later does not mutate this local ndarray.
                frame = self._frame
                use_zone = self._frame_uses_zone
                display_transform = self._frame_display_transform
                source_generation = self._source_generation
                self._processed_frame_id = self._frame_id
                self._frame_ready.clear()

            try:
                started_at = time.perf_counter()
                thresholds = self._get_thresholds()
                detections = self._detect(
                    frame,
                    thresholds,
                    use_zone=use_zone,
                    display_transform=display_transform,
                )
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                self.inference_fps = 1.0 / elapsed
                elapsed_ms = elapsed * 1000.0
                self._inference_count += 1
                self._inference_ms = elapsed_ms
                if self._inference_ms_avg <= 0:
                    self._inference_ms_avg = elapsed_ms
                else:
                    self._inference_ms_avg = (
                        self._inference_ms_avg * 0.8 + elapsed_ms * 0.2
                    )
                last_inference_at = time.monotonic()
            except Exception as error:
                self.error_ready.emit(f"YOLO error: {error}")
                time.sleep(0.1)
                continue

            with self._lock:
                if source_generation != self._source_generation:
                    # The source was closed, seeked, or replaced while this
                    # inference was running. Never publish stale boxes.
                    continue
                self.last_detections = detections
            self.result_ready.emit(
                self._format_result(detections, thresholds, use_zone=use_zone)
            )

    @property
    def performance_stats(self):
        """Return detector counters without touching the model thread."""
        with self._lock:
            return {
                "inference_count": self._inference_count,
                "inference_ms": self._inference_ms,
                "inference_ms_avg": self._inference_ms_avg,
                "inference_fps": self.inference_fps,
                "source_frames_skipped": self._source_frames_skipped,
                "device": self.device or "unknown",
            }

    def _update_zone(self, shape):
        now = time.monotonic()
        if now - self._zone_last_check < FILE_CHECK_INTERVAL_S:
            return
        self._zone_last_check = now

        try:
            mtime = ZONE_FILE.stat().st_mtime_ns
        except FileNotFoundError:
            with self._lock:
                self.zone = None
                self.zone_mtime = 0
            return

        with self._lock:
            reload_needed = self.zone is None or mtime != self.zone_mtime
        if reload_needed:
            try:
                zone = self.load_zone(shape)
            except OSError:
                zone = None
            with self._lock:
                self.zone = zone
                self.zone_mtime = mtime

    def _get_thresholds(self):
        now = time.monotonic()
        if now - self._thresholds_last_check < FILE_CHECK_INTERVAL_S:
            return self._thresholds
        self._thresholds_last_check = now

        try:
            mtime = DISTANCE_SETTINGS_FILE.stat().st_mtime_ns
        except OSError:
            mtime = 0
        if mtime != self._thresholds_mtime:
            self._thresholds = load_distance_thresholds()
            self._thresholds_mtime = mtime
        return self._thresholds

    @staticmethod
    def _map_model_box_to_display(box, transform):
        """Map a model-letterboxed box to the independently displayed frame."""
        model = transform["model"]
        display = transform["display"]
        model_scale = max(float(model["scale"]), 1e-9)
        display_scale = float(display["scale"])

        def map_x(value):
            source_x = (float(value) - float(model["offset_x"])) / model_scale
            return int(round(source_x * display_scale + float(display["offset_x"])))

        def map_y(value):
            source_y = (float(value) - float(model["offset_y"])) / model_scale
            return int(round(source_y * display_scale + float(display["offset_y"])))

        x1, y1, x2, y2 = box
        return map_x(x1), map_y(y1), map_x(x2), map_y(y2)

    def _detect(self, frame, thresholds, use_zone=True, display_transform=None):
        danger_dist = thresholds["danger"]
        warning_dist = thresholds["warning"]

        if frame.shape[:2] == (MODEL_IMGSZ, MODEL_IMGSZ):
            # CaptureThread already prepared the exact model input. Avoid a
            # second 640x640 allocation/copy on every inference cycle.
            model_frame = frame
            transform = {
                "scale": 1.0,
                "offset_x": 0,
                "offset_y": 0,
                "source_w": MODEL_IMGSZ,
                "source_h": MODEL_IMGSZ,
            }
        else:
            model_frame, transform = letterbox_with_meta(
                frame, (MODEL_IMGSZ, MODEL_IMGSZ)
            )
        if use_zone:
            self._update_zone(model_frame.shape)
        results = self._predict(model_frame)

        detections = []
        zone = self.get_zone() if use_zone else None
        # Objects off to the side project onto a shorter "straight-ahead"
        # depth than their true (slant) distance from the camera; correct
        # for that using the horizontal offset from the frame's center.
        principal_x = model_frame.shape[1] / 2.0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = _normalize_label(self.model.names[class_id])
                # Ignore custom-model classes without distance calibration.
                if label not in REAL_HEIGHT:
                    continue
                score = float(box.conf[0])
                if score < self.conf:
                    continue

                model_box = tuple(map(int, box.xyxy[0]))
                x1, y1, x2, y2 = model_box
                box_height = max(y2 - y1, 1)
                if box_height < MIN_BOX_HEIGHT:
                    continue

                bottom_center = ((x1 + x2) // 2, y2)
                if zone is not None and cv2.pointPolygonTest(zone, bottom_center, False) < 0:
                    continue

                object_height = self._object_heights.get(
                    label, DEFAULT_OBJECT_HEIGHTS.get(label, 1.0)
                )
                depth = (object_height * self._focal_length) / box_height
                box_center_x = (x1 + x2) / 2.0
                lateral_ratio = (box_center_x - principal_x) / self._focal_length
                raw_distance = depth * (1 + lateral_ratio ** 2) ** 0.5
                display_box = (
                    self._map_model_box_to_display(model_box, display_transform)
                    if display_transform is not None
                    else unletterbox_box(model_box, transform)
                )
                detection = {
                    "box": display_box,
                    "label": label,
                    "score": round(score, 2),
                    "raw_dist": raw_distance,
                }
                detections.append(detection)

        detections = self._smooth_detections(detections, danger_dist, warning_dist)
        return sorted(detections, key=lambda detection: detection["dist"])

    def _smooth_detections(self, detections, danger_dist, warning_dist):
        updated = []
        used_previous = set()
        now = time.monotonic()

        for detection in detections:
            previous_index, previous = self._find_previous_detection(
                detection, used_previous, now
            )

            if previous is not None:
                smoothed_distance = (
                    SMOOTHING_ALPHA * detection["raw_dist"]
                    + (1.0 - SMOOTHING_ALPHA) * previous["raw_dist"]
                )
                previous_status = previous["status"]
                velocity = self._estimate_velocity(previous, detection, now)
                used_previous.add(previous_index)
            else:
                smoothed_distance = detection["raw_dist"]
                previous_status = None
                velocity = (0.0, 0.0)

            status = self._classify_distance(
                smoothed_distance,
                danger_dist,
                warning_dist,
                previous_status,
            )
            style = STATUS_STYLE[status]

            updated_detection = {
                "box": detection["box"],
                "label": detection["label"],
                "score": detection["score"],
                "raw_dist": smoothed_distance,
                "dist": round(smoothed_distance, 2),
                "status": status,
                "status_thai": style["thai"],
                "color": style["color"],
                "html": style["html"],
                "timestamp": now,
                "velocity": velocity,
            }
            updated.append(updated_detection)

        self._track_memory = updated
        return updated

    @staticmethod
    def _estimate_velocity(previous, detection, now):
        """Smoothed box-center velocity in px/sec, used to predict a fast
        object's position between inference results (see _predict_box)."""
        previous_timestamp = previous.get("timestamp")
        if previous_timestamp is None:
            return (0.0, 0.0)

        dt = now - previous_timestamp
        if dt <= 1e-3:
            return previous.get("velocity", (0.0, 0.0))

        px1, py1, px2, py2 = previous["box"]
        x1, y1, x2, y2 = detection["box"]
        raw_vx = ((x1 + x2) / 2.0 - (px1 + px2) / 2.0) / dt
        raw_vy = ((y1 + y2) / 2.0 - (py1 + py2) / 2.0) / dt

        prev_vx, prev_vy = previous.get("velocity", (0.0, 0.0))
        vx = VELOCITY_SMOOTHING_ALPHA * raw_vx + (1.0 - VELOCITY_SMOOTHING_ALPHA) * prev_vx
        vy = VELOCITY_SMOOTHING_ALPHA * raw_vy + (1.0 - VELOCITY_SMOOTHING_ALPHA) * prev_vy
        vx = max(-MAX_TRACK_SPEED_PX_S, min(MAX_TRACK_SPEED_PX_S, vx))
        vy = max(-MAX_TRACK_SPEED_PX_S, min(MAX_TRACK_SPEED_PX_S, vy))
        return (vx, vy)

    def _find_previous_detection(self, detection, used_previous, now):
        best_index = None
        best_previous = None
        best_iou = 0.0

        for index, previous in enumerate(self._track_memory):
            if index in used_previous:
                continue
            if previous["label"] != detection["label"]:
                continue

            # Match against where this track is predicted to be *now*, not
            # where it was last seen -- otherwise a fast-moving object
            # quickly slides out of raw IoU range and the track is dropped
            # (losing hysteresis/velocity continuity) even though it's the
            # same object.
            predicted_box = self._predict_box(previous, now)
            iou = _box_iou(predicted_box, detection["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = index
                best_previous = previous

        if best_iou >= TRACK_IOU_THRESHOLD:
            return best_index, best_previous
        return None, None

    @staticmethod
    def _classify_distance(distance, danger_dist, warning_dist, previous_status=None):
        if previous_status == "DANGER" and distance <= danger_dist + DANGER_HYSTERESIS_M:
            return "DANGER"
        if previous_status == "WARNING":
            if distance <= danger_dist:
                return "DANGER"
            if distance <= warning_dist + WARNING_HYSTERESIS_M:
                return "WARNING"

        if distance <= danger_dist:
            return "DANGER"
        if distance <= warning_dist:
            return "WARNING"
        return "SAFE"

    @staticmethod
    def _format_result(detections, thresholds, use_zone=True):
        if not detections and not use_zone:
            return '<span style="color:#94a3b8;">ไม่พบวัตถุในภาพ</span>'
        if not detections:
            return '<span style="color:#94a3b8;">ไม่พบวัตถุในโซนตรวจจับ</span>'

        alert_detections = [
            detection for detection in detections
            if detection["status"] in ("DANGER", "WARNING", "SAFE")
        ]
        if not alert_detections:
            return '<span style="color:#94a3b8;">ยังไม่มีวัตถุในระยะเตือน</span>'

        header = (
            f'<span style="color:#e5e7eb;">'
            f'อันตราย ≤ {thresholds["danger"]} M | '
            f'ระวัง ≤ {thresholds["warning"]} M | '
            f'ปลอดภัย > {thresholds["warning"]} M'
            f'</span><br><br>'
        )
        rows = "".join(
            f'<span style="color:{detection["html"]};">'
            f'{index}. {LABEL_THAI.get(detection["label"], detection["label"])} '
            f'{detection["dist"]:.0f} เมตร '
            f'[{detection["status_thai"]}] '
            f'conf {detection["score"]:.2f}'
            f'</span><br>'
            for index, detection in enumerate(alert_detections, 1)
        )
        return header + rows

    def stop(self, wait=True, timeout_ms=10000):
        self._stop_requested.set()
        self._frame_ready.set()
        if not wait or not self.isRunning():
            return True
        return self.wait(timeout_ms)

    def update_thresholds(self, danger, warning):
        """Apply values saved by the UI without waiting for the file poll."""
        with self._lock:
            self._thresholds = {"danger": float(danger), "warning": float(warning)}
        self._thresholds_last_check = time.monotonic()

    def update_model_thresholds(self, conf, iou):
        """Apply YOLO conf/IoU values without reloading the model."""
        values = normalize_model_thresholds(conf, iou)
        with self._lock:
            self.conf = values["conf"]
            self.iou = values["iou"]
        return values

    def update_camera_settings(self, focal_length):
        """Apply camera calibration without restarting the detection thread."""
        with self._lock:
            self._focal_length = float(focal_length)
            self._track_memory = []

    def update_object_heights(self, object_heights):
        """Apply per-class real-world heights without restarting detection."""
        with self._lock:
            self._object_heights = {
                label: float(value)
                for label, value in object_heights.items()
                if label in DEFAULT_OBJECT_HEIGHTS
            }
            for label, default in DEFAULT_OBJECT_HEIGHTS.items():
                self._object_heights.setdefault(label, default)
            self._track_memory = []

    def invalidate_zone(self):
        """Request a zone reload on the next inference frame."""
        self._zone_last_check = 0.0
