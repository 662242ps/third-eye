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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye")
        self.resize(1400, 800)

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
        except Exception as error:
            self.alert.setText(f"Cannot load YOLO model: {error}")

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

        title = QLabel("Third Eye - Object Detection and Distance Warning")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        self.distance_status = QLabel("")
        self.distance_status.setStyleSheet("font-size:13px; color:#333;")

        self.btn_distance = QPushButton("Distance Setting")
        self.btn_distance.setFixedSize(165, 45)
        self.btn_distance.setStyleSheet("background:#C9C9C9; border-radius:8px;")
        self.btn_distance.clicked.connect(self.open_distance_setting)

        self.btn_setting = QPushButton("Zone Setting")
        self.btn_setting.setFixedSize(140, 45)
        self.btn_setting.setStyleSheet("background:#C9C9C9; border-radius:8px;")
        self.btn_setting.clicked.connect(self.open_zone_setting)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.distance_status)
        header_layout.addWidget(self.btn_distance)
        header_layout.addWidget(self.btn_setting)

        control = QFrame()
        control.setFixedHeight(90)
        control.setStyleSheet("background:#5A5A5A;")
        control_layout = QHBoxLayout(control)
        control_layout.setContentsMargins(40, 0, 0, 0)
        control_layout.setSpacing(20)

        self.btn_open = QPushButton("Open Camera")
        self.btn_close = QPushButton("Close Camera")
        self.btn_video = QPushButton("Upload Video")
        for button in (self.btn_open, self.btn_close, self.btn_video):
            button.setFixedSize(220, 50)
            button.setStyleSheet("background:#D0D0D0; border-radius:10px; font-size:16px;")
            control_layout.addWidget(button)
        control_layout.addStretch()
        self.btn_open.clicked.connect(self.open_camera)
        self.btn_close.clicked.connect(lambda: self.close_camera())
        self.btn_video.clicked.connect(self.open_video)

        content = QFrame()
        content.setStyleSheet("background:#5A5A5A;")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(40, 20, 40, 20)
        content_layout.setSpacing(30)

        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setMinimumSize(900, 520)
        self.video.setStyleSheet("background:black;")
        self.alert = QTextEdit()
        self.alert.setReadOnly(True)
        self.alert.setMinimumWidth(320)
        self.alert.setStyleSheet("background:black; color:yellow; font-size:18px;")
        content_layout.addWidget(self.video)
        content_layout.addWidget(self.alert)

        main.addWidget(header)
        main.addWidget(control)
        main.addWidget(content)

    def update_distance_status(self):
        thresholds = load_distance_thresholds()
        self.distance_status.setText(
            f"Danger <= {thresholds['danger']} m | "
            f"Warning <= {thresholds['warning']} m | "
            f"Safe > {thresholds['warning']} m"
        )

    def open_camera(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.alert.setText("Cannot open camera")
            return
        self.last_time = time.time()
        self.timer.start(33)
        self.alert.setText("Camera opened")

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
            self.alert.setText("Cannot open video")
            return
        self.last_time = time.time()
        self.timer.start(33)
        self.alert.setText("Video opened")

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
            self.alert.setText("Camera closed")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.close_camera()
            self.alert.setText("Video ended or camera frame unavailable")
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
            cv2.polylines(frame, [zone], True, (0, 0, 255), 2)

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
            f"{self.last_info}<br><span style='color:white;'>FPS: {self.fps:.1f}</span>"
        )

    def on_yolo_result(self, info):
        self.last_info = info

    def on_yolo_error(self, message):
        self.last_info = f"<span style='color:red;'>{message}</span>"

    def open_distance_setting(self):
        self.distance_win = DistanceSettingWindow()
        self.distance_win.thresholds_saved.connect(self.on_thresholds_saved)
        self.distance_win.show()

    def on_thresholds_saved(self, danger, warning):
        self.update_distance_status()
        self.last_info = (
            f"<span style='color:white;'>Distance setting saved: "
            f"Danger <= {danger} M, Warning <= {warning} M</span>"
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
