import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.camera_setting_window import CameraSettingWindow
from ui.distance_setting_window import DistanceSettingWindow
from ui.zone_setting_window import ZoneSettingWindow
from vision.camera_config import load_camera_settings
from vision.distance_config import load_distance_thresholds
from vision.yolo_thread import YoloThread


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "tmp"

APP_STYLE = """
QMainWindow {
    background: #0b1120;
}
QFrame#header {
    background: #101827;
    border-bottom: 1px solid #26364c;
}
QFrame#toolbar {
    background: #0b1120;
}
QFrame#content {
    background: #0b1120;
}
QFrame#videoCard, QFrame#alertCard {
    background: #121d2f;
    border: 1px solid #2b3a52;
    border-radius: 16px;
}
QFrame#metricCard {
    background: #101a2b;
    border: 1px solid #26364c;
    border-radius: 12px;
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
QPushButton:disabled {
    background: #1e293b;
    color: #64748b;
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
QMenu {
    background: #172033;
    color: #f8fafc;
    border: 1px solid #334155;
    padding: 6px;
}
QMenu::item {
    border-radius: 7px;
    padding: 9px 28px 9px 12px;
}
QMenu::item:selected {
    background: #2563eb;
}
QTextEdit {
    background: #020617;
    color: #facc15;
    border: 1px solid #243244;
    border-radius: 14px;
    padding: 12px;
    font-size: 14px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye")
        self.resize(1120, 720)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(APP_STYLE)

        self.cap = None
        self.current_frame = None
        self.last_time = time.time()
        self.fps = 0.0
        self.source_name = "No source"
        self.last_info = ""
        self.yolo = None
        self.zone_win = None
        self.distance_win = None
        self.camera_win = None
        self.test_frame = None
        self.test_mode = False

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
        header.setFixedHeight(84)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 7, 28, 7)
        header_layout.setSpacing(14)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("Third Eye")
        title.setFont(QFont("Arial", 25, QFont.Bold))
        title.setMinimumHeight(43)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        subtitle = QLabel("ระบบตรวจจับวัตถุบนถนน วัดระยะ และแจ้งเตือนความเสี่ยง")
        subtitle.setMinimumHeight(24)
        subtitle.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        subtitle.setStyleSheet(
            "color:#94a3b8; font-family:'Leelawadee UI'; font-size:13px;"
        )
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.status_badge = QLabel("กำลังเริ่มระบบ")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedWidth(138)
        self.status_badge.setStyleSheet(
            "background:#334155; color:white; border-radius:12px; padding:8px; font-weight:700;"
        )

        header_layout.addLayout(title_box)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)

        toolbar = QFrame(objectName="toolbar")
        toolbar.setFixedHeight(66)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(28, 12, 28, 12)
        toolbar_layout.setSpacing(14)

        self.btn_open = QPushButton("เปิดกล้อง")
        self.btn_open.setObjectName("successButton")
        self.btn_close = QPushButton("ปิดกล้อง")
        self.btn_close.setObjectName("dangerButton")
        self.btn_video = QPushButton("อัปโหลดวิดีโอ")
        self.btn_video.setObjectName("secondaryButton")
        self.btn_test = QPushButton("Test รูปภาพ")
        self.btn_test.setObjectName("secondaryButton")
        self.btn_settings = QPushButton("ตั้งค่า")
        self.btn_settings.setObjectName("settingButton")
        self.btn_settings.setToolTip("การตั้งค่า")
        self.btn_settings.setFont(QFont("Arial", 15, QFont.Bold))

        for button in (self.btn_open, self.btn_close, self.btn_video, self.btn_test):
            button.setFixedSize(142, 42)
            toolbar_layout.addWidget(button)

        toolbar_layout.addStretch()
        self.btn_settings.setFixedSize(120, 42)
        toolbar_layout.addWidget(self.btn_settings)

        self.btn_open.clicked.connect(self.open_camera)
        self.btn_close.clicked.connect(lambda: self.close_camera())
        self.btn_video.clicked.connect(self.open_video)
        self.btn_test.clicked.connect(self.open_test_image)
        settings_menu = QMenu(self)
        distance_action = settings_menu.addAction("ตั้งค่าระยะ")
        zone_action = settings_menu.addAction("ตั้งค่าโซน")
        camera_action = settings_menu.addAction("ตั้งค่ากล้อง")
        distance_action.triggered.connect(self.open_distance_setting)
        zone_action.triggered.connect(self.open_zone_setting)
        camera_action.triggered.connect(self.open_camera_setting)
        self.btn_settings.setMenu(settings_menu)

        content = QFrame(objectName="content")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(24, 10, 24, 24)
        content_layout.setSpacing(18)

        video_card = QFrame(objectName="videoCard")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(10)

        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setMinimumSize(480, 380)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet(
            "background:#020617; border:1px solid #334155; border-radius:14px; color:#64748b;"
        )
        self.video.setText("ยังไม่ได้เปิดกล้องหรือวิดีโอ")
        video_layout.addWidget(self.video)

        alert_card = QFrame(objectName="alertCard")
        alert_card.setMinimumWidth(320)
        alert_card.setMaximumWidth(370)
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

        self.source_metric = self._make_metric("SOURCE", self.source_name)
        self.fps_metric = self._make_metric("DISPLAY / AI FPS", "--")
        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        metrics.addWidget(self.source_metric)
        metrics.addWidget(self.fps_metric)
        alert_layout.addLayout(metrics)

        self.distance_status = QLabel("")
        self.distance_status.setWordWrap(True)
        self.distance_status.setStyleSheet(
            "background:#0b1220; color:#aab9cc; border-radius:10px; "
            "padding:9px; font-size:12px;"
        )
        alert_layout.addWidget(self.distance_status)
        alert_layout.addWidget(self.alert)

        content_layout.addWidget(video_card)
        content_layout.addWidget(alert_card)

        main.addWidget(header)
        main.addWidget(toolbar)
        main.addWidget(content)

    @staticmethod
    def _make_metric(title, value):
        card = QFrame(objectName="metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        label = QLabel(title)
        label.setStyleSheet("color:#718096; font-size:10px; font-weight:700;")
        metric_value = QLabel(value)
        metric_value.setObjectName("metricValue")
        metric_value.setStyleSheet("color:#e5e7eb; font-size:13px; font-weight:700;")
        metric_value.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(metric_value)
        card.value_label = metric_value
        return card

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
        self.test_mode = False
        self.test_frame = None
        camera_index = load_camera_settings()["camera_index"]
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.alert.setText("ไม่สามารถเปิดกล้องได้")
            self.set_status("กล้องผิดพลาด", "#ef4444")
            return
        self.last_time = time.time()
        self.source_name = f"Camera {camera_index}"
        self.source_metric.value_label.setText(self.source_name)
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
        self.test_mode = False
        self.test_frame = None
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.alert.setText("ไม่สามารถเปิดวิดีโอได้")
            self.set_status("วิดีโอผิดพลาด", "#ef4444")
            return
        self.last_time = time.time()
        self.source_name = Path(path).name
        self.source_metric.value_label.setText(self.source_name)
        self.timer.start(33)
        self.alert.setText("เปิดวิดีโอแล้ว")
        self.set_status("กำลังทำงาน", "#16a34a")

    def open_test_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "เลือกภาพสำหรับ Test",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if not path:
            return

        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.alert.setText("ไม่สามารถเปิดไฟล์ภาพนี้ได้")
            self.set_status("เปิดภาพไม่สำเร็จ", "#ef4444")
            return

        self.close_camera(clear_display=False)
        self.test_mode = True
        self.test_frame = cv2.resize(image, (640, 640))
        self.current_frame = self.test_frame.copy()
        self.source_name = Path(path).name
        self.source_metric.value_label.setText(self.source_name)
        self.fps_metric.value_label.setText("กำลังตรวจจับ...")
        self.last_info = (
            "<span style='color:#94a3b8;'>กำลังตรวจจับวัตถุจากภาพเต็ม "
            "โดยไม่ใช้โซน...</span>"
        )
        self.alert.setHtml(self.last_info)
        self._display_frame(self.test_frame.copy(), draw_zone=False)
        self.set_status("กำลัง Test ภาพ", "#2563eb")

        if self.yolo is None:
            self.alert.setText("โมเดลยังไม่พร้อมใช้งาน")
            self.set_status("โมเดลไม่พร้อม", "#ef4444")
            return
        self._submit_test_frame()

    def _submit_test_frame(self):
        if self.test_mode and self.test_frame is not None and self.yolo is not None:
            if not self.yolo.update_frame(self.test_frame, use_zone=False):
                QTimer.singleShot(50, self._submit_test_frame)

    def close_camera(self, clear_display=True):
        self.timer.stop()
        self.test_mode = False
        self.test_frame = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.source_name = "No source"
        self.source_metric.value_label.setText(self.source_name)
        self.fps_metric.value_label.setText("--")
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

        frame = cv2.resize(frame, (640, 640))
        self.current_frame = frame.copy()
        if self.yolo is not None:
            self.yolo.update_frame(frame)

        self._display_frame(frame, draw_zone=True)

    def _display_frame(self, frame, draw_zone):
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

        zone = self.yolo.get_zone() if draw_zone and self.yolo else None
        if zone is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (30, 80, 180))
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
            cv2.polylines(frame, [zone], True, (56, 189, 248), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        self.video.setPixmap(
            QPixmap.fromImage(image).scaled(self.video.size(), Qt.KeepAspectRatio)
        )
        if draw_zone:
            now = time.time()
            self.fps = 1.0 / max(now - self.last_time, 1e-6)
            self.last_time = now
            ai_fps = self.yolo.inference_fps if self.yolo is not None else 0.0
            self.fps_metric.value_label.setText(f"{self.fps:.1f} / {ai_fps:.1f}")
            self.alert.setHtml(self.last_info)

    def on_yolo_result(self, info):
        self.last_info = info
        if self.test_mode and self.test_frame is not None:
            self._display_frame(self.test_frame.copy(), draw_zone=False)
            self.alert.setHtml(info)
            ai_fps = self.yolo.inference_fps if self.yolo is not None else 0.0
            self.fps_metric.value_label.setText(f"AI {ai_fps:.1f} FPS")
            self.set_status("Test เสร็จแล้ว", "#22c55e")

    def on_yolo_error(self, message):
        self.last_info = f"<span style='color:#ef4444;'>{message}</span>"
        if self.test_mode:
            self.alert.setHtml(self.last_info)
        self.set_status("ตรวจจับผิดพลาด", "#ef4444")

    def open_distance_setting(self):
        self.distance_win = DistanceSettingWindow()
        self.distance_win.thresholds_saved.connect(self.on_thresholds_saved)
        self.distance_win.show()

    def open_camera_setting(self):
        self.camera_win = CameraSettingWindow()
        self.camera_win.settings_saved.connect(self.on_camera_settings_saved)
        self.camera_win.show()

    def on_camera_settings_saved(self, camera_index, focal_length):
        if self.yolo is not None:
            self.yolo.update_camera_settings(focal_length)
        self.last_info = (
            f"<span style='color:#e5e7eb;'>บันทึกค่ากล้องแล้ว: "
            f"Camera {camera_index}, Focal Length {focal_length:.1f} px</span>"
        )
        self.alert.setHtml(self.last_info)

    def on_thresholds_saved(self, danger, warning):
        self.update_distance_status()
        if self.yolo is not None:
            self.yolo.update_thresholds(danger, warning)
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
        if self.yolo is not None:
            self.zone_win.zone_saved.connect(self.yolo.invalidate_zone)
        self.zone_win.show()

    def closeEvent(self, event):
        self.close_camera()
        if self.yolo is not None and self.yolo.isRunning():
            if not self.yolo.stop():
                self.alert.setText("กำลังรอให้ระบบตรวจจับหยุดทำงาน...")
                event.ignore()
                QTimer.singleShot(250, self.close)
                return
        event.accept()
