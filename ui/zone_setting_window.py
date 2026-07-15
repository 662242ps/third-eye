import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt


ZONE_DIR = "zones"
ACTIVE_ZONE = os.path.join(ZONE_DIR, "active.txt")

VIEW_W = 1000
VIEW_H = 600


class ZoneSettingWindow(QMainWindow):
    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("Third Eye - Zone Setting")
        self.resize(1400, 800)

        os.makedirs(ZONE_DIR, exist_ok=True)

        # -------------------------
        # Load background image
        # -------------------------
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
        else:
            img = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)

        self.image = cv2.resize(img, (VIEW_W, VIEW_H))

        # -------------------------
        # Default zone (triangle) – pixel (UI space)
        # -------------------------
        self.zone_points = [
            [100, 550],
            [500, 180],
            [900, 550]
        ]
        self.drag_index = None

        # ================= UI =================
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # ---------- Header ----------
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet("background:#E6E6E6;")
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 0, 20, 0)

        title = QLabel("Third Eye โปรแกรมทดสอบตรวจจับและวัดระยะ")
        title.setFont(QFont("Arial", 18, QFont.Bold))

        h.addWidget(title)
        h.addStretch()

        # ---------- Content ----------
        content = QFrame()
        content.setStyleSheet("background:#6A6A6A;")
        c = QHBoxLayout(content)
        c.setContentsMargins(30, 20, 30, 20)
        c.setSpacing(30)

        # ----- Image panel -----
        self.image_label = QLabel()
        self.image_label.setFixedSize(VIEW_W, VIEW_H)
        self.image_label.setStyleSheet("background:black;")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        # ----- Right panel -----
        right = QVBoxLayout()
        right.setSpacing(15)

        lbl = QLabel("Zone Preset")
        lbl.setFont(QFont("Arial", 14))

        self.preset_box = QComboBox()

        self.preset_box.blockSignals(True)
        self.load_presets()
        self.preset_box.blockSignals(False)

        self.preset_box.currentTextChanged.connect(self.load_zone)

        btn_submit = QPushButton("Submit (Set Active)")
        btn_submit.setFixedSize(200, 55)
        btn_submit.setStyleSheet(
            "background:#E0E0E0; border-radius:10px; font-size:16px;"
        )
        btn_submit.clicked.connect(self.save_zone)

        hint = QLabel("ลากจุดสีขาวเพื่อปรับ Zone")
        hint.setStyleSheet("color:black;")

        right.addWidget(lbl)
        right.addWidget(self.preset_box)
        right.addWidget(hint)
        right.addStretch()
        right.addWidget(btn_submit, alignment=Qt.AlignCenter)

        c.addWidget(self.image_label)
        c.addLayout(right)

        main.addWidget(header)
        main.addWidget(content)

        self.update_view()

    # ================= Drawing =================
    def update_view(self):
        frame = self.image.copy()
        pts = np.array(self.zone_points, np.int32)

        cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
        for x, y in self.zone_points:
            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, VIEW_W, VIEW_H, 3 * VIEW_W, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    # ================= Mouse =================
    def mouse_press(self, event):
        x, y = event.pos().x(), event.pos().y()
        for i, (px, py) in enumerate(self.zone_points):
            if abs(px - x) < 15 and abs(py - y) < 15:
                self.drag_index = i
                break

    def mouse_move(self, event):
        if self.drag_index is not None:
            self.zone_points[self.drag_index] = [
                max(0, min(event.pos().x(), VIEW_W)),
                max(0, min(event.pos().y(), VIEW_H))
            ]
            self.update_view()

    def mouse_release(self, event):
        self.drag_index = None

    # ================= Preset =================
    def load_presets(self):
        self.preset_box.clear()

        files = [
            f for f in os.listdir(ZONE_DIR)
            if f.endswith(".txt") and os.path.isfile(os.path.join(ZONE_DIR, f))
        ]

        if not files:
            default = "default.txt"
            self.save_file(os.path.join(ZONE_DIR, default))
            files = [default]

        self.preset_box.addItems(files)

    def load_zone(self, name):
        if not name or not name.endswith(".txt"):
            return

        path = os.path.join(ZONE_DIR, name)
        if not os.path.isfile(path):
            return

        pts = []
        with open(path) as f:
            for line in f:
                rx, ry = map(float, line.strip().split(","))
                pts.append([int(rx * VIEW_W), int(ry * VIEW_H)])

        if len(pts) >= 3:
            self.zone_points = pts
            self.update_view()

    # ================= Save =================
    def save_file(self, path):
        with open(path, "w") as f:
            for x, y in self.zone_points:
                rx = x / VIEW_W
                ry = y / VIEW_H
                f.write(f"{rx:.6f},{ry:.6f}\n")

    def save_zone(self):
        name = self.preset_box.currentText()
        if not name.endswith(".txt"):
            return

        path = os.path.join(ZONE_DIR, name)

        self.save_file(path)
        self.save_file(ACTIVE_ZONE)

        self.preset_box.blockSignals(True)
        self.load_presets()
        self.preset_box.setCurrentText(name)
        self.preset_box.blockSignals(False)

        print("Active zone set:", path)


# -------------------------
# Run directly (for test)
# -------------------------
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = ZoneSettingWindow("road.jpg")
    win.show()
    sys.exit(app.exec_())
