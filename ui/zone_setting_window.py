import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZONE_DIR = PROJECT_ROOT / "zones"
ACTIVE_ZONE = ZONE_DIR / "active.txt"
VIEW_W = 1000
VIEW_H = 600


class ZoneSettingWindow(QMainWindow):
    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("Third Eye - Zone Setting")
        self.resize(1400, 800)
        ZONE_DIR.mkdir(exist_ok=True)

        image_file = Path(image_path) if image_path else None
        image = cv2.imread(str(image_file)) if image_file and image_file.is_file() else None
        if image is None:
            image = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        self.image = cv2.resize(image, (VIEW_W, VIEW_H))

        self.zone_points = [[100, 550], [500, 180], [900, 550]]
        self.drag_index = None
        self._build_ui()
        self.load_active_zone()
        self.update_view()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet("background:#E6E6E6;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        title = QLabel("Third Eye - Zone Setting")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()

        content = QFrame()
        content.setStyleSheet("background:#6A6A6A;")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(30, 20, 30, 20)
        content_layout.setSpacing(30)

        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setFixedSize(VIEW_W, VIEW_H)
        self.image_label.setStyleSheet("background:black;")
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        right = QVBoxLayout()
        right.setSpacing(15)
        label = QLabel("Zone Preset")
        label.setFont(QFont("Arial", 14))
        self.preset_box = QComboBox()
        self.preset_box.blockSignals(True)
        self.load_presets()
        self.preset_box.blockSignals(False)
        self.preset_box.currentTextChanged.connect(self.load_zone)

        submit = QPushButton("Submit (Set Active)")
        submit.setFixedSize(200, 55)
        submit.setStyleSheet("background:#E0E0E0; border-radius:10px; font-size:16px;")
        submit.clicked.connect(self.save_zone)
        hint = QLabel("Drag the white points to adjust the zone")

        right.addWidget(label)
        right.addWidget(self.preset_box)
        right.addWidget(hint)
        right.addStretch()
        right.addWidget(submit, alignment=Qt.AlignCenter)
        content_layout.addWidget(self.image_label)
        content_layout.addLayout(right)

        main.addWidget(header)
        main.addWidget(content)

    def update_view(self):
        frame = self.image.copy()
        points = np.array(self.zone_points, np.int32)
        cv2.polylines(frame, [points], True, (0, 0, 255), 3)
        for x, y in self.zone_points:
            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb_frame.data, VIEW_W, VIEW_H, 3 * VIEW_W, QImage.Format_RGB888
        ).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def mouse_press(self, event):
        x, y = event.pos().x(), event.pos().y()
        for index, (point_x, point_y) in enumerate(self.zone_points):
            if abs(point_x - x) < 15 and abs(point_y - y) < 15:
                self.drag_index = index
                break

    def mouse_move(self, event):
        if self.drag_index is not None:
            self.zone_points[self.drag_index] = [
                max(0, min(event.pos().x(), VIEW_W - 1)),
                max(0, min(event.pos().y(), VIEW_H - 1)),
            ]
            self.update_view()

    def mouse_release(self, event):
        self.drag_index = None

    def load_presets(self):
        self.preset_box.clear()
        presets = sorted(
            path.name
            for path in ZONE_DIR.glob("*.txt")
            if path.name != ACTIVE_ZONE.name and path.is_file()
        )
        if not presets:
            self.save_file(ZONE_DIR / "default.txt")
            presets = ["default.txt"]
        self.preset_box.addItems(presets)

    @staticmethod
    def read_zone(path):
        points = []
        try:
            with Path(path).open(encoding="utf-8") as file:
                for line in file:
                    x_ratio, y_ratio = map(float, line.strip().split(","))
                    if not (0 <= x_ratio <= 1 and 0 <= y_ratio <= 1):
                        return None
                    points.append([int(x_ratio * VIEW_W), int(y_ratio * VIEW_H)])
        except (OSError, ValueError):
            return None
        return points if len(points) >= 3 else None

    def load_active_zone(self):
        points = self.read_zone(ACTIVE_ZONE)
        if points:
            self.zone_points = points

    def load_zone(self, name):
        if not name.endswith(".txt"):
            return
        points = self.read_zone(ZONE_DIR / name)
        if points:
            self.zone_points = points
            self.update_view()

    def save_file(self, path):
        with Path(path).open("w", encoding="utf-8") as file:
            for x, y in self.zone_points:
                file.write(f"{x / VIEW_W:.6f},{y / VIEW_H:.6f}\n")

    def save_zone(self):
        name = self.preset_box.currentText()
        if not name.endswith(".txt"):
            return
        self.save_file(ZONE_DIR / name)
        self.save_file(ACTIVE_ZONE)
        self.preset_box.blockSignals(True)
        self.load_presets()
        self.preset_box.setCurrentText(name)
        self.preset_box.blockSignals(False)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ZoneSettingWindow()
    window.show()
    sys.exit(app.exec_())
