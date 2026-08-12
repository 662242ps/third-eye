import os
import time
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Lock

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from vision.camera_config import load_camera_settings
from vision.distance_config import DISTANCE_SETTINGS_FILE, load_distance_thresholds

# Resizing/drawing is lightweight here. Let Torch own the CPU worker pool instead
# of allowing OpenCV and Torch to oversubscribe the same cores.
cv2.setNumThreads(1)

try:
    import torch
except ImportError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "best.pt"
ZONE_FILE = PROJECT_ROOT / "zones" / "active.txt"
FILE_CHECK_INTERVAL_S = 0.75

# Approximate real-world object heights in meters.
REAL_HEIGHT = {
    "person": 1.65,
    "car": 1.50,
    "truck": 2.80,
    "motorcycle": 1.20,
    "bus": 3.10,
}

CLASS_ALIASES = {
    "motorbike": "motorcycle",
    "bike": "motorcycle",
}

# Practical confidence thresholds for real webcam/video use.
# Higher values reduce false positives. Lower values detect more objects but may be noisy.
conf = 0.5
CLASS_CONF = {
    "person": conf,
    "car": conf,
    "truck": conf,
    "motorcycle": conf,
    
}

# Keep inference input identical to the square image size used for training.
MODEL_IMGSZ = 640
MODEL_IOU = 0.8
# Road scenes rarely have more than a handful of relevant objects in frame;
# capping detections lower trims NMS/post-processing work.
MODEL_MAX_DET = 20

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


def _select_device():
    if torch is not None and torch.cuda.is_available():
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

    def __init__(self, model_path=None):
        super().__init__()

        self.model_path = Path(model_path) if model_path else MODEL_FILE
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Heavy model loading is deferred to run() so the Qt UI remains
        # responsive while the model/device is initialized.
        self.device = None
        self.use_half = False
        self.model = None
        self.class_ids = []
        self.model_ready = False

        self._stop_requested = Event()
        self._frame_ready = Event()
        self._lock = Lock()
        self._frame = None
        self._frame_uses_zone = True
        self._frame_id = 0
        self._processed_frame_id = 0

        self.zone = None
        self.zone_mtime = 0
        self._zone_last_check = 0.0
        self.last_detections = []
        self._track_memory = []
        self.inference_fps = 0.0
        self._thresholds = load_distance_thresholds()
        self._focal_length = load_camera_settings()["focal_length"]
        self._thresholds_mtime = 0
        self._thresholds_last_check = 0.0

    def _initialize_model(self):
        self.device = _select_device()
        self.use_half = self.device == "cuda"
        if torch is not None:
            torch.set_grad_enabled(False)
            if not self.use_half:
                cpu_threads = max(1, min(4, os.cpu_count() or 2))
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
        self._warm_up()
        self.model_ready = True

    def _warm_up(self):
        """Build the inference graph before the first real camera frame."""
        try:
            self.model(
                np.zeros((MODEL_IMGSZ, MODEL_IMGSZ, 3), dtype=np.uint8),
                imgsz=MODEL_IMGSZ,
                classes=self.class_ids or None,
                half=self.use_half,
                verbose=False,
            )
        except Exception as error:
            print(f"YOLO warm-up skipped: {error}")

    def update_frame(self, frame, use_zone=True):
        """Store the newest frame and discard older unprocessed frames."""
        with self._lock:
            if self._stop_requested.is_set():
                return False
            if self._frame_id != self._processed_frame_id:
                return False
            self._frame = frame.copy()
            self._frame_uses_zone = bool(use_zone)
            self._frame_id += 1
        self._frame_ready.set()
        return True

    def clear_frame(self):
        """Pause inference when no camera/video source is active."""
        with self._lock:
            self._frame = None
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

    def get_zone(self):
        with self._lock:
            return None if self.zone is None else self.zone.copy()

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
        while not self._stop_requested.is_set():
            self._frame_ready.wait(timeout=0.1)
            self._frame_ready.clear()

            with self._lock:
                if self._frame is None or self._frame_id == self._processed_frame_id:
                    continue
                # update_frame already owns a private copy; replacing self._frame
                # later does not mutate this local ndarray.
                frame = self._frame
                use_zone = self._frame_uses_zone
                self._processed_frame_id = self._frame_id

            try:
                started_at = time.perf_counter()
                if use_zone:
                    self._update_zone(frame.shape)
                thresholds = self._get_thresholds()
                detections = self._detect(frame, thresholds, use_zone=use_zone)
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                self.inference_fps = 1.0 / elapsed
            except Exception as error:
                self.error_ready.emit(f"YOLO error: {error}")
                time.sleep(0.1)
                continue

            with self._lock:
                self.last_detections = detections
            self.result_ready.emit(
                self._format_result(detections, thresholds, use_zone=use_zone)
            )

    def _update_zone(self, shape):
        now = time.monotonic()
        if now - self._zone_last_check < FILE_CHECK_INTERVAL_S:
            return
        self._zone_last_check = now

        try:
            mtime = ZONE_FILE.stat().st_mtime
        except FileNotFoundError:
            with self._lock:
                self.zone = None
                self.zone_mtime = 0
            return

        with self._lock:
            reload_needed = self.zone is None or mtime != self.zone_mtime
        if reload_needed:
            zone = self.load_zone(shape)
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

    def _detect(self, frame, thresholds, use_zone=True):
        danger_dist = thresholds["danger"]
        warning_dist = thresholds["warning"]

        min_conf = min(CLASS_CONF.values())
        inference_context = torch.inference_mode() if torch is not None else nullcontext()
        with inference_context:
            results = self.model(
                frame,
                conf=min_conf,
                iou=MODEL_IOU,
                imgsz=MODEL_IMGSZ,
                max_det=MODEL_MAX_DET,
                classes=self.class_ids or None,
                half=self.use_half,
                augment=False,
                verbose=False,
            )

        detections = []
        zone = self.get_zone() if use_zone else None
        # Objects off to the side project onto a shorter "straight-ahead"
        # depth than their true (slant) distance from the camera; correct
        # for that using the horizontal offset from the frame's center.
        principal_x = frame.shape[1] / 2.0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = _normalize_label(self.model.names[class_id])
                # Ignore custom-model classes without distance calibration.
                if label not in REAL_HEIGHT:
                    continue
                score = float(box.conf[0])
                if score < CLASS_CONF.get(label, 1.0):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_height = max(y2 - y1, 1)
                if box_height < MIN_BOX_HEIGHT:
                    continue

                bottom_center = ((x1 + x2) // 2, y2)
                if zone is not None and cv2.pointPolygonTest(zone, bottom_center, False) < 0:
                    continue

                depth = (REAL_HEIGHT[label] * self._focal_length) / box_height
                box_center_x = (x1 + x2) / 2.0
                lateral_ratio = (box_center_x - principal_x) / self._focal_length
                raw_distance = depth * (1 + lateral_ratio ** 2) ** 0.5
                detection = {
                    "box": (x1, y1, x2, y2),
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
            if detection["status"] in ("DANGER", "WARNING")
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

    def stop(self):
        self._stop_requested.set()
        self._frame_ready.set()
        return self.wait(10000)

    def update_thresholds(self, danger, warning):
        """Apply values saved by the UI without waiting for the file poll."""
        with self._lock:
            self._thresholds = {"danger": float(danger), "warning": float(warning)}
        self._thresholds_last_check = time.monotonic()

    def update_camera_settings(self, focal_length):
        """Apply camera calibration without restarting the detection thread."""
        with self._lock:
            self._focal_length = float(focal_length)
            self._track_memory = []

    def invalidate_zone(self):
        """Request a zone reload on the next inference frame."""
        self._zone_last_check = 0.0
