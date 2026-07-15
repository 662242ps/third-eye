import time
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.distance_setting_window import DistanceSettingWindow
from ui.zone_setting_window import ZoneSettingWindow
from vision.distance_config import load_distance_thresholds
from vision.yolo_thread import YoloThread


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "tmp"

APP_STYLE = """
QMainWindow {
    background: #0f172a;
}
QFrame#header {
    background: #111827;
    border-bottom: 1px solid #263143;
}
QFrame#toolbar {
    background: #0f172a;
    border-bottom: 1px solid #1f2937;
}
QFrame#content {
    background: #0f172a;
}
QFrame#videoCard, QFrame#alertCard {
    background: #172033;
    border: 1px solid #2d3a4f;
    border-radius: 18px;
}
QLabel {
    color: #e5e7eb;
}
QPushButton {
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 15px;
    font-weight: 700;
}
QPushButton#successButton {
    background: #16a34a;
    color: white;
}
QPushButton#successButton:hover {
    background: #15803d;
}
QPushButton#dangerButton {
    background: #dc2626;
    color: white;
}
QPushButton#dangerButton:hover {
    background: #b91c1c;
}
QPushButton#secondaryButton {
    background: #334155;
    color: #f8fafc;
}
QPushButton#secondaryButton:hover {
    background: #475569;
}
QPushButton#settingButton {
    background: #2563eb;
    color: white;
}
QPushButton#settingButton:hover {
    background: #1d4ed8;
}
QTextEdit {
    background: #020617;
    color: #facc15;
    border: 1px solid #243244;
    border-radius: 14px;
    padding: 12px;
    font-size: 16px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye")
        self.resize(1440, 850)
        self.setStyleSheet(APP_STYLE)

        self.cap = None
        self.current_frame = None
        self.last_time = time.time()
        self.fps = 0.0
        self.last_info = ""
        self.yolo = None
        self.zone_win = None
        self.distance_win = None

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_distance_status()

        try:
            self.yolo = YoloThread()
            self.yolo.result_ready.connect(self.on_yolo_result)
            self.yolo.error_ready.connect(self.on_yolo_error)
            self.yolo.start()
            self.set_status("พร้อมใช้งาน", "#22c55e")
        except Exception as error:
            self.alert.setText(f"Cannot load YOLO model: {error}")
            self.set_status("โหลดโมเดลไม่สำเร็จ", "#ef4444")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        header = QFrame(objectName="header")
        header.setFixedHeight(88)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 12, 28, 12)
        header_layout.setSpacing(14)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Third Eye")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        subtitle = QLabel("ระบบตรวจจับวัตถุบนถนน วัดระยะ และแจ้งเตือนความเสี่ยง")
        subtitle.setStyleSheet("color:#94a3b8; font-size:13px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.distance_status = QLabel("")
        self.distance_status.setStyleSheet(
            "background:#0b1220; color:#cbd5e1; border:1px solid #243244; "
            "border-radius:12px; padding:8px 12px; font-size:13px;"
        )

        self.status_badge = QLabel("กำลังเริ่มระบบ")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedWidth(130)
        self.status_badge.setStyleSheet(
            "background:#334155; color:white; border-radius:12px; padding:8px; font-weight:700;"
        )

        self.btn_distance = QPushButton("ตั้งค่าระยะ")
        self.btn_distance.setObjectName("settingButton")
        self.btn_distance.setFixedSize(140, 46)
        self.btn_distance.clicked.connect(self.open_distance_setting)

        self.btn_setting = QPushButton("ตั้งค่าโซน")
        self.btn_setting.setObjectName("settingButton")
        self.btn_setting.setFixedSize(130, 46)
        self.btn_setting.clicked.connect(self.open_zone_setting)

        header_layout.addLayout(title_box)
        header_layout.addStretch()
        header_layout.addWidget(self.distance_status)
        header_layout.addWidget(self.status_badge)
        header_layout.addWidget(self.btn_distance)
        header_layout.addWidget(self.btn_setting)

        toolbar = QFrame(objectName="toolbar")
        toolbar.setFixedHeight(92)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(28, 18, 28, 18)
        toolbar_layout.setSpacing(14)

        self.btn_open = QPushButton("เปิดกล้อง")
        self.btn_open.setObjectName("successButton")
        self.btn_close = QPushButton("ปิดกล้อง")
        self.btn_close.setObjectName("dangerButton")
        self.btn_video = QPushButton("อัปโหลดวิดีโอ")
        self.btn_video.setObjectName("secondaryButton")

        for button in (self.btn_open, self.btn_close, self.btn_video):
            button.setFixedSize(180, 50)
            toolbar_layout.addWidget(button)

        toolbar_layout.addStretch()
        self.btn_open.clicked.connect(self.open_camera)
        self.btn_close.clicked.connect(lambda: self.close_camera())
        self.btn_video.clicked.connect(self.open_video)

        content = QFrame(objectName="content")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(28, 24, 28, 28)
        content_layout.setSpacing(22)

        video_card = QFrame(objectName="videoCard")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(10)

        video_title = QLabel("มุมมองกล้อง / วิดีโอ")
        video_title.setFont(QFont("Arial", 14, QFont.Bold))
        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setMinimumSize(900, 560)
        self.video.setStyleSheet(
            "background:#020617; border:1px solid #334155; border-radius:14px; color:#64748b;"
        )
        self.video.setText("ยังไม่ได้เปิดกล้องหรือวิดีโอ")
        video_layout.addWidget(video_title)
        video_layout.addWidget(self.video)

        alert_card = QFrame(objectName="alertCard")
        alert_card.setFixedWidth(360)
        alert_layout = QVBoxLayout(alert_card)
        alert_layout.setContentsMargins(14, 14, 14, 14)
        alert_layout.setSpacing(10)

        alert_title = QLabel("รายการแจ้งเตือน")
        alert_title.setFont(QFont("Arial", 14, QFont.Bold))
        self.alert = QTextEdit()
        self.alert.setReadOnly(True)
        self.alert.setHtml(
            "<span style='color:#94a3b8;'>เปิดกล้องหรืออัปโหลดวิดีโอเพื่อเริ่มตรวจจับ</span>"
        )
        alert_layout.addWidget(alert_title)
        alert_layout.addWidget(self.alert)

        content_layout.addWidget(video_card)
        content_layout.addWidget(alert_card)

        main.addWidget(header)
        main.addWidget(toolbar)
        main.addWidget(content)

    def set_status(self, text, color):
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"background:{color}; color:white; border-radius:12px; padding:8px; font-weight:700;"
        )

    def update_distance_status(self):
        thresholds = load_distance_thresholds()
        self.distance_status.setText(
            f"อันตราย ≤ {thresholds['danger']} m | "
            f"ระวัง ≤ {thresholds['warning']} m | "
            f"ปลอดภัย > {thresholds['warning']} m"
        )

    def open_camera(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.alert.setText("ไม่สามารถเปิดกล้องได้")
            self.set_status("กล้องผิดพลาด", "#ef4444")
            return
        self.last_time = time.time()
        self.timer.start(33)
        self.alert.setText("เปิดกล้องแล้ว")
        self.set_status("กำลังทำงาน", "#16a34a")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return

        self.close_camera(clear_display=False)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.alert.setText("ไม่สามารถเปิดวิดีโอได้")
            self.set_status("วิดีโอผิดพลาด", "#ef4444")
            return
        self.last_time = time.time()
        self.timer.start(33)
        self.alert.setText("เปิดวิดีโอแล้ว")
        self.set_status("กำลังทำงาน", "#16a34a")

    def close_camera(self, clear_display=True):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        if self.yolo is not None:
            self.yolo.clear_frame()
        if clear_display:
            self.video.clear()
            self.video.setText("ยังไม่ได้เปิดกล้องหรือวิดีโอ")
            self.alert.setText("ปิดกล้องแล้ว")
            self.set_status("พร้อมใช้งาน", "#22c55e")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.close_camera()
            self.alert.setText("วิดีโอจบแล้ว หรือไม่สามารถอ่านเฟรมจากกล้องได้")
            return

        frame = cv2.resize(frame, (800, 600))
        self.current_frame = frame.copy()
        if self.yolo is not None:
            self.yolo.update_frame(frame)

        for detection in self.yolo.get_detections() if self.yolo else []:
            x1, y1, x2, y2 = detection["box"]
            color = detection["color"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f'{detection["label"]} {detection["dist"]} M [{detection["status"]}]',
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        zone = self.yolo.get_zone() if self.yolo else None
        if zone is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (30, 80, 180))
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
            cv2.polylines(frame, [zone], True, (56, 189, 248), 2)

        now = time.time()
        self.fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        self.video.setPixmap(
            QPixmap.fromImage(image).scaled(self.video.size(), Qt.KeepAspectRatio)
        )
        self.alert.setHtml(
            f"{self.last_info}<br><span style='color:#cbd5e1;'>FPS: {self.fps:.1f}</span>"
        )

    def on_yolo_result(self, info):
        self.last_info = info

    def on_yolo_error(self, message):
        self.last_info = f"<span style='color:#ef4444;'>{message}</span>"
        self.set_status("ตรวจจับผิดพลาด", "#ef4444")

    def open_distance_setting(self):
        self.distance_win = DistanceSettingWindow()
        self.distance_win.thresholds_saved.connect(self.on_thresholds_saved)
        self.distance_win.show()

    def on_thresholds_saved(self, danger, warning):
        self.update_distance_status()
        self.last_info = (
            f"<span style='color:#e5e7eb;'>บันทึกระยะแล้ว: "
            f"อันตราย ≤ {danger} M, ระวัง ≤ {warning} M</span>"
        )

    def open_zone_setting(self):
        image_path = None
        TEMP_DIR.mkdir(exist_ok=True)
        if self.current_frame is not None:
            image_path = TEMP_DIR / "zone_preview.jpg"
            cv2.imwrite(str(image_path), self.current_frame)

        self.zone_win = ZoneSettingWindow(str(image_path) if image_path else None)
        self.zone_win.show()

    def closeEvent(self, event):
        self.close_camera()
        if self.yolo is not None and self.yolo.isRunning():
            self.yolo.stop()
        event.accept()
