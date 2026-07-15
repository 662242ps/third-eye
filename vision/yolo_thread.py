import time
from pathlib import Path
from threading import Event, Lock

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


FOCAL_LENGTH = 600
REAL_HEIGHT = {
    "person": 1.7,
    "car": 1.8,
    "truck": 4.0,
    "motorcycle": 1.4,
}

CLASS_CONF = {
    "person": 0.01,
    "car": 0.35,
    "truck": 0.75,
    "motorcycle": 0.2,
}

DANGER_DIST = 5.0
WARNING_DIST = 15.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "best.pt"
ZONE_FILE = PROJECT_ROOT / "zones" / "active.txt"


class YoloThread(QThread):
    """Runs inference only for the most recent camera frame."""

    result_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        if not MODEL_FILE.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

        self.model = YOLO(str(MODEL_FILE))
        self.model.to("cpu")
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
        class_names = (
            self.model.names.items()
            if isinstance(self.model.names, dict)
            else enumerate(self.model.names)
        )
        self.class_ids = [
            class_id
            for class_id, name in class_names
            if name.lower() in REAL_HEIGHT
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
        results = self.model(
            frame,
            conf=min(CLASS_CONF.values()),
            iou=0.35,
            classes=self.class_ids or None,
            verbose=False,
        )
        detections = []
        zone = self.get_zone()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id].lower()
                score = float(box.conf[0])
                if score < CLASS_CONF.get(label, 1.0):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bottom_center = ((x1 + x2) // 2, y2)
                if zone is not None and cv2.pointPolygonTest(zone, bottom_center, False) < 0:
                    continue

                distance = round(
                    (REAL_HEIGHT[label] * FOCAL_LENGTH) / max(y2 - y1, 1), 2
                )
                if distance <= DANGER_DIST:
                    status, color, html = "DANGER", (0, 0, 255), "red"
                elif distance <= WARNING_DIST:
                    status, color, html = "WARNING", (0, 165, 255), "orange"
                else:
                    status, color, html = "SAFE", (0, 255, 0), "lime"

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "dist": distance,
                        "status": status,
                        "color": color,
                        "html": html,
                    }
                )

        return sorted(detections, key=lambda detection: detection["dist"])

    @staticmethod
    def _format_result(detections):
        if not detections:
            return '<span style="color:gray;">No object</span>'
        return "".join(
            f'<span style="color:{detection["html"]};">'
            f'{index}. {detection["label"]} {detection["dist"]} M '
            f'[{detection["status"]}]</span><br>'
            for index, detection in enumerate(detections, 1)
        )

    def stop(self):
        self._stop_requested.set()
        self._frame_ready.set()
        self.wait(3000)
