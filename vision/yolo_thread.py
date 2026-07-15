import time
import cv2
import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

# Distance parameters

FOCAL_LENGTH = 600
REAL_HEIGHT = {
    "person": 1.7,
    "car": 1.8,
    "truck": 4.0,
    "motorcycle": 1.4
}

# Per-class confidence
CLASS_CONF = {
    "person": 0.01,
    "car": 0.35,
    "truck": 0.75,
    "motorcycle": 0.2
}

# Warning levels (meters)

DANGER_DIST = 5.0
WARNING_DIST = 15.0

ZONE_FILE = "zones/active.txt"


class YoloThread(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.model = YOLO("models/best.pt")
        self.model.to("cpu")

        print("MODEL CLASSES:", self.model.names)

        self.running = True
        self.frame = None
        self.skip = 2
        self.count = 0

        self.zone = None
        self.zone_mtime = 0
        self.last_detections = []

    # รับ frame จาก UI
    def update_frame(self, frame):
        self.frame = frame

    # โหลด zone (ratio → pixel)
    def load_zone(self, shape):
        if not os.path.isfile(ZONE_FILE):
            return None

        h, w = shape[:2]
        pts = []

        with open(ZONE_FILE) as f:
            for line in f:
                try:
                    rx, ry = map(float, line.strip().split(","))
                    pts.append((int(rx * w), int(ry * h)))
                except:
                    pass

        return np.array(pts, dtype=np.int32) if len(pts) >= 3 else None

    def get_zone(self):
        return None if self.zone is None else self.zone.copy()

    # Thread main loop
    def run(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.001)
                continue

            self.count += 1
            if self.count % self.skip != 0:
                continue

            frame = self.frame.copy()

            # reload zone (เฉพาะตอนเปลี่ยน)
            if os.path.isfile(ZONE_FILE):
                mtime = os.path.getmtime(ZONE_FILE)
                if self.zone is None or mtime != self.zone_mtime:
                    self.zone = self.load_zone(frame.shape)
                    self.zone_mtime = mtime

            results = self.model(
                frame,
                conf=0.15,             
                iou=0.35,
                classes=[0, 1, 2, 3],     
                verbose=False
            )

            detections = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id].lower()
                    score = float(box.conf[0])

                    # กรอง conf แยกคลาส
                    if score < CLASS_CONF.get(label, 0.2):
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox_h = y2 - y1

                    # bottom-center
                    cx = (x1 + x2) // 2
                    cy = y2

                    # zone filter
                    if self.zone is not None:
                        if cv2.pointPolygonTest(self.zone, (cx, cy), False) < 0:
                            continue

                    real_h = REAL_HEIGHT.get(label, 1.7)
                    dist = round(
                        (real_h * FOCAL_LENGTH) / max(bbox_h, 1),
                        2
                    )

                    if dist <= DANGER_DIST:
                        status = "DANGER"
                        color = (0, 0, 255)
                        html = "red"
                    elif dist <= WARNING_DIST:
                        status = "WARNING"
                        color = (0, 165, 255)
                        html = "orange"
                    else:
                        status = "SAFE"
                        color = (0, 255, 0)
                        html = "lime"

                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "dist": dist,
                        "status": status,
                        "color": color,
                        "html": html
                    })

            detections.sort(key=lambda d: d["dist"])
            self.last_detections = detections

            # ส่งข้อความให้ UI
            if detections:
                info = ""
                for i, d in enumerate(detections, 1):
                    info += (
                        f'<span style="color:{d["html"]};">'
                        f'{i}. {d["label"]} {d["dist"]} M [{d["status"]}]'
                        f'</span><br>'
                    )
            else:
                info = '<span style="color:gray;">No object</span>'

            self.result_ready.emit(info)

    def stop(self):
        self.running = False
        self.wait()
