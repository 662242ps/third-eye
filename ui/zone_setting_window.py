import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
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
DEFAULT_ZONE = ZONE_DIR / "default.txt"
VIEW_W = 1000
VIEW_H = 600

APP_STYLE = """
QMainWindow {
    background: #0f172a;
}
QWidget { font-family: 'Leelawadee UI'; }
QLabel {
    color: #e5e7eb;
}
QFrame#header {
    background: #111827;
    border-bottom: 1px solid #263143;
}
QFrame#contentCard {
    background: #172033;
    border: 1px solid #2d3a4f;
    border-radius: 18px;
}
QFrame#sideCard {
    background: #101827;
    border: 1px solid #243244;
    border-radius: 16px;
}
QComboBox {
    background: #0b1220;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 15px;
}
QComboBox:hover {
    border: 1px solid #38bdf8;
}
QPushButton#primaryButton {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 16px;
    font-weight: 700;
}
QPushButton#primaryButton:hover {
    background: #1d4ed8;
}
QPushButton#primaryButton:pressed {
    background: #1e40af;
}
"""


class ZoneSettingWindow(QMainWindow):
    zone_saved = pyqtSignal()

    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("Third Eye - ตั้งค่าโซนตรวจจับ")
        self.resize(1360, 780)
        self.setStyleSheet(APP_STYLE)
        ZONE_DIR.mkdir(exist_ok=True)

        image_file = Path(image_path) if image_path else None
        image = cv2.imread(str(image_file)) if image_file and image_file.is_file() else None
        if image is None:
            image = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        self.image = cv2.resize(image, (VIEW_W, VIEW_H))

        self.zone_points = [[100, 550], [500, 180], [900, 550]]
        self.drag_index = None
        self.status = None
        self._build_ui()
        self.load_active_zone()
        self.update_view()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        header = QFrame(objectName="header")
        header.setFixedHeight(82)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(28, 12, 28, 12)
        header_layout.setSpacing(2)
        title = QLabel("ตั้งค่าโซนตรวจจับ")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        subtitle = QLabel("ลากจุดสีขาวเพื่อกำหนดพื้นที่ที่ต้องการตรวจจับ จากนั้นกดบันทึก")
        subtitle.setStyleSheet("color:#94a3b8; font-size:13px;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        outer = QFrame()
        outer_layout = QHBoxLayout(outer)
        outer_layout.setContentsMargins(26, 26, 26, 26)
        outer_layout.setSpacing(22)

        content_card = QFrame(objectName="contentCard")
        content_layout = QHBoxLayout(content_card)
        content_layout.setContentsMargins(18, 18, 18, 18)
        content_layout.setSpacing(20)

        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setFixedSize(VIEW_W, VIEW_H)
        self.image_label.setStyleSheet(
            "background:#020617; border:1px solid #334155; border-radius:14px;"
        )
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        side_card = QFrame(objectName="sideCard")
        side_card.setFixedWidth(260)
        right = QVBoxLayout(side_card)
        right.setContentsMargins(18, 18, 18, 18)
        right.setSpacing(14)

        label = QLabel("Zone Preset")
        label.setFont(QFont("Arial", 14, QFont.Bold))
        hint = QLabel("เลือก preset หรือปรับจุดบนภาพแล้วบันทึกเป็นโซนใช้งาน")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8; font-size:13px;")

        self.preset_box = QComboBox()
        self.preset_box.blockSignals(True)
        self.load_presets()
        self.preset_box.blockSignals(False)
        self.preset_box.currentTextChanged.connect(self.load_zone)

        instruction = QLabel(
            "วิธีใช้\n"
            "1. ลากจุดสีขาว 3 จุด\n"
            "2. ให้ครอบพื้นที่ถนน\n"
            "3. กดบันทึกโซน"
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            "background:#0b1220; color:#cbd5e1; border-radius:12px; "
            "padding:12px; font-size:13px;"
        )

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#facc15; font-size:13px;")

        submit = QPushButton("บันทึกโซนนี้")
        submit.setObjectName("primaryButton")
        submit.setFixedHeight(48)
        submit.clicked.connect(self.save_zone)

        right.addWidget(label)
        right.addWidget(hint)
        right.addWidget(self.preset_box)
        right.addWidget(instruction)
        right.addWidget(self.status)
        right.addStretch()
        right.addWidget(submit)

        content_layout.addWidget(self.image_label)
        content_layout.addWidget(side_card)
        outer_layout.addWidget(content_card)

        main.addWidget(header)
        main.addWidget(outer)

    def update_view(self):
        frame = self.image.copy()
        overlay = frame.copy()
        points = np.array(self.zone_points, np.int32)
        cv2.fillPoly(overlay, [points], (30, 80, 180))
        frame = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)
        cv2.polylines(frame, [points], True, (56, 189, 248), 3)
        for index, (x, y) in enumerate(self.zone_points, 1):
            cv2.circle(frame, (x, y), 12, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 12, (56, 189, 248), 2)
            cv2.putText(
                frame,
                str(index),
                (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (15, 23, 42),
                2,
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb_frame.data, VIEW_W, VIEW_H, 3 * VIEW_W, QImage.Format_RGB888
        ).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def mouse_press(self, event):
        x, y = event.pos().x(), event.pos().y()
        for index, (point_x, point_y) in enumerate(self.zone_points):
            if abs(point_x - x) < 18 and abs(point_y - y) < 18:
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
        if len(points) < 3:
            return None
        polygon = np.array(points, dtype=np.int32)
        if len({tuple(point) for point in points}) != len(points):
            return None
        if abs(cv2.contourArea(polygon)) < 100.0:
            return None
        return points

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
            if self.status:
                self.status.setStyleSheet("color:#38bdf8; font-size:13px;")
                self.status.setText(f"โหลด preset: {name}")

    def save_file(self, path):
        with Path(path).open("w", encoding="utf-8") as file:
            for x, y in self.zone_points:
                file.write(f"{x / VIEW_W:.6f},{y / VIEW_H:.6f}\n")

    def save_zone(self):
        name = self.preset_box.currentText()
        if not name.endswith(".txt"):
            return
        polygon = np.array(self.zone_points, dtype=np.int32)
        if (
            len({tuple(point) for point in self.zone_points}) != len(self.zone_points)
            or abs(cv2.contourArea(polygon)) < 100.0
        ):
            self.status.setStyleSheet("color:#ef4444; font-size:13px; font-weight:700;")
            self.status.setText("ไม่สามารถบันทึกได้: โซนต้องมีอย่างน้อย 3 จุดและมีพื้นที่")
            return

        if (ZONE_DIR / name) != DEFAULT_ZONE:
            self.save_file(ZONE_DIR / name)
        self.save_file(ACTIVE_ZONE)
        self.preset_box.blockSignals(True)
        self.load_presets()
        self.preset_box.setCurrentText(name)
        self.preset_box.blockSignals(False)
        self.status.setStyleSheet("color:#22c55e; font-size:13px; font-weight:700;")
        self.status.setText("บันทึกสำเร็จ ใช้โซนนี้แล้ว")
        self.zone_saved.emit()
        self.close()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ZoneSettingWindow()
    window.show()
    sys.exit(app.exec_())
