import os
import cv2
import time

from PyQt5.QtWidgets import (
    QMainWindow, QWidget,
    QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout,
    QFrame, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

from ui.zone_setting_window import ZoneSettingWindow
from vision.yolo_thread import YoloThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye")
        self.resize(1400, 800)

        # runtime
        self.cap = None

        # fps
        self.last_time = time.time()
        self.fps = 0.0

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

        self.btn_setting = QPushButton("ตั้งค่า ⚙")
        self.btn_setting.setFixedSize(120, 45)
        self.btn_setting.setStyleSheet("background:#C9C9C9; border-radius:8px;")
        self.btn_setting.clicked.connect(self.open_zone_setting)

        h.addWidget(title)
        h.addStretch()
        h.addWidget(self.btn_setting)

        # ---------- Control ----------
        control = QFrame()
        control.setFixedHeight(90)
        control.setStyleSheet("background:#5A5A5A;")
        c = QHBoxLayout(control)
        c.setContentsMargins(40, 0, 0, 0)
        c.setSpacing(20)

        self.btn_open = QPushButton("Open Camera")
        self.btn_close = QPushButton("Close Camera")
        self.btn_video = QPushButton("Upload Video")

        for b in (self.btn_open, self.btn_close, self.btn_video):
            b.setFixedSize(220, 50)
            b.setStyleSheet(
                "background:#D0D0D0; border-radius:10px; font-size:16px;"
            )

        self.btn_open.clicked.connect(self.open_camera)
        self.btn_close.clicked.connect(self.close_camera)
        self.btn_video.clicked.connect(self.open_video)

        c.addWidget(self.btn_open)
        c.addWidget(self.btn_close)
        c.addWidget(self.btn_video)
        c.addStretch()

        # ---------- Content ----------
        content = QFrame()
        content.setStyleSheet("background:#5A5A5A;")
        ct = QHBoxLayout(content)
        ct.setContentsMargins(40, 20, 40, 20)
        ct.setSpacing(30)

        self.video = QLabel()
        self.video.setMinimumSize(900, 520)
        self.video.setStyleSheet("background:black;")
        self.video.setAlignment(Qt.AlignCenter)

        self.alert = QTextEdit()
        self.alert.setReadOnly(True)
        self.alert.setMinimumWidth(320)
        self.alert.setStyleSheet(
            "background:black; color:yellow; font-size:18px;"
        )

        ct.addWidget(self.video)
        ct.addWidget(self.alert)

        main.addWidget(header)
        main.addWidget(control)
        main.addWidget(content)

        # ---------- Timer ----------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 🔥 YOLO Thread (ใหม่)
        self.yolo = YoloThread()
        self.yolo.result_ready.connect(self.on_yolo_result)
        self.yolo.start()

        self.last_info = ""

    # ================= Camera / Video =================
    def open_camera(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.alert.setText("❌ Cannot open camera")
            self.cap = None
            return
        self.timer.start(33)   # ~30 FPS คงที่ → ลื่น
        self.alert.setText("Camera opened")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi)"
        )
        if not path:
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.timer.start(33)
        self.alert.setText("Video opened")

    def close_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.alert.setText("Camera closed")
            self.video.clear()

    # ================= Loop =================
    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # 🔥 เปลี่ยนตรงนี้
        frame = cv2.resize(frame, (800, 600))

        # ส่ง frame ให้ YOLO คิด
        self.yolo.update_frame(frame)

        # วาด detection
        for d in self.yolo.last_detections:
            x1, y1, x2, y2 = d["box"]
            color = d["color"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f'{d["label"]} {d["dist"]} M [{d["status"]}]',
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # วาด zone
        zone = self.yolo.get_zone()
        if zone is not None:
            cv2.polylines(frame, [zone], True, (0, 0, 255), 2)

        # FPS
        now = time.time()
        self.fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)

        self.video.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video.size(),
                Qt.KeepAspectRatio
            )
        )

        self.alert.setHtml(
            f"{self.last_info}<br>"
            f"<span style='color:white;'>FPS: {self.fps:.1f}</span>"
        )


    # ================= YOLO Result =================
    def on_yolo_result(self, info):
        self.last_info = info

    # ================= Zone Setting =================
    def open_zone_setting(self):
        image_path = None
        os.makedirs("tmp", exist_ok=True)

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                image_path = "tmp/zone_preview.jpg"
                cv2.imwrite(image_path, frame)

        self.zone_win = ZoneSettingWindow(image_path)
        self.zone_win.show()

    # ================= Close =================
    def closeEvent(self, e):
        self.timer.stop()
        if self.yolo:
            self.yolo.stop()
        if self.cap:
            self.cap.release()
        e.accept()
