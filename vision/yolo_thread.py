import time
from pathlib import Path
from threading import Event, Lock

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from vision.distance_config import load_distance_thresholds

try:
    import torch
except ImportError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "best.pt"
ZONE_FILE = PROJECT_ROOT / "zones" / "active.txt"

# Camera/distance calibration
# FOCAL_LENGTH should be calibrated with a known object distance for best accuracy.
FOCAL_LENGTH = 600

# Approximate real-world object heights in meters.
REAL_HEIGHT = {
    "person": 1.70,
    "car": 1.55,
    "truck": 3.00,
    "motorcycle": 1.25,
}

# Practical confidence thresholds for real webcam/video use.
# Higher values reduce false positives. Lower values detect more objects but may be noisy.
CLASS_CONF = {
    "person": 0.30,
    "car": 0.40,
    "truck": 0.55,
    "motorcycle": 0.35,
}

MODEL_IMGSZ = 640
MODEL_IOU = 0.45
MODEL_MAX_DET = 30

MIN_BOX_HEIGHT = 18
SMOOTHING_ALPHA = 0.35
TRACK_IOU_THRESHOLD = 0.30
DANGER_HYSTERESIS_M = 0.60
WARNING_HYSTERESIS_M = 1.00

STATUS_STYLE = {
    "DANGER": {"color": (0, 0, 255), "html": "#ef4444", "thai": "อันตราย"},
    "WARNING": {"color": (0, 165, 255), "html": "#f59e0b", "thai": "ระวัง"},
    "SAFE": {"color": (0, 255, 0), "html": "#22c55e", "thai": "ปลอดภัย"},
}


def _select_device():
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_label(name):
    return str(name).strip().lower()


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

    def __init__(self):
        super().__init__()

        if not MODEL_FILE.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

        self.device = _select_device()
        self.model = YOLO(str(MODEL_FILE))
        self.model.to(self.device)
        print(f"YOLO device: {self.device}")
        print("MODEL CLASSES:", self.model.names)

        self._stop_requested = Event()
        self._frame_ready = Event()
        self._lock = Lock()
        self._frame = None
        self._frame_id = 0
        self._processed_frame_id = 0

        self.zone = None
        self.zone_mtime = 0
        self.last_detections = []
        self._track_memory = []

        class_names = (
            self.model.names.items()
            if isinstance(self.model.names, dict)
            else enumerate(self.model.names)
        )
        self.class_ids = [
            class_id
            for class_id, name in class_names
            if _normalize_label(name) in REAL_HEIGHT
        ]

    def update_frame(self, frame):
        """Store the newest frame and discard older unprocessed frames."""
        with self._lock:
            self._frame = frame.copy()
            self._frame_id += 1
        self._frame_ready.set()

    def clear_frame(self):
        """Pause inference when no camera/video source is active."""
        with self._lock:
            self._frame = None
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

    def get_detections(self):
        with self._lock:
            return [detection.copy() for detection in self.last_detections]

    def run(self):
        while not self._stop_requested.is_set():
            self._frame_ready.wait(timeout=0.1)
            self._frame_ready.clear()

            with self._lock:
                if self._frame is None or self._frame_id == self._processed_frame_id:
                    continue
                frame = self._frame.copy()
                self._processed_frame_id = self._frame_id

            try:
                self._update_zone(frame.shape)
                detections = self._detect(frame)
            except Exception as error:
                self.error_ready.emit(f"YOLO error: {error}")
                time.sleep(0.1)
                continue

            with self._lock:
                self.last_detections = detections
            self.result_ready.emit(self._format_result(detections))

    def _update_zone(self, shape):
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

    def _detect(self, frame):
        thresholds = load_distance_thresholds()
        danger_dist = thresholds["danger"]
        warning_dist = thresholds["warning"]

        min_conf = min(CLASS_CONF.values())
        results = self.model(
            frame,
            conf=min_conf,
            iou=MODEL_IOU,
            imgsz=MODEL_IMGSZ,
            max_det=MODEL_MAX_DET,
            classes=self.class_ids or None,
            verbose=False,
        )

        detections = []
        zone = self.get_zone()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = _normalize_label(self.model.names[class_id])
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

                raw_distance = (REAL_HEIGHT[label] * FOCAL_LENGTH) / box_height
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

        for detection in detections:
            previous_index, previous = self._find_previous_detection(detection, used_previous)

            if previous is not None:
                smoothed_distance = (
                    SMOOTHING_ALPHA * detection["raw_dist"]
                    + (1.0 - SMOOTHING_ALPHA) * previous["raw_dist"]
                )
                previous_status = previous["status"]
                used_previous.add(previous_index)
            else:
                smoothed_distance = detection["raw_dist"]
                previous_status = None

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
            }
            updated.append(updated_detection)

        self._track_memory = updated
        return updated

    def _find_previous_detection(self, detection, used_previous):
        best_index = None
        best_previous = None
        best_iou = 0.0

        for index, previous in enumerate(self._track_memory):
            if index in used_previous:
                continue
            if previous["label"] != detection["label"]:
                continue

            iou = _box_iou(previous["box"], detection["box"])
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
    def _format_result(detections):
        if not detections:
            return '<span style="color:#94a3b8;">ไม่พบวัตถุในโซนตรวจจับ</span>'

        thresholds = load_distance_thresholds()
        header = (
            f'<span style="color:#e5e7eb;">'
            f'อันตราย ≤ {thresholds["danger"]} M | '
            f'ระวัง ≤ {thresholds["warning"]} M | '
            f'ปลอดภัย > {thresholds["warning"]} M'
            f'</span><br><br>'
        )
        rows = "".join(
            f'<span style="color:{detection["html"]};">'
            f'{index}. {detection["label"]} '
            f'{detection["dist"]} M '
            f'[{detection["status_thai"]}] '
            f'conf {detection["score"]:.2f}'
            f'</span><br>'
            for index, detection in enumerate(detections, 1)
        )
        return header + rows

    def stop(self):
        self._stop_requested.set()
        self._frame_ready.set()
        self.wait(3000)
