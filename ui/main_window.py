import csv
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
    QSlider,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.alert_setting_window import AlertSettingWindow
from ui.camera_setting_window import CameraSettingWindow
from ui.distance_setting_window import DistanceSettingWindow
from ui.model_setting_window import ModelSettingWindow
from ui.zone_setting_window import ZoneSettingWindow
from vision.alert_config import load_alert_settings
from vision.alert_sound import DangerAlarm
from vision.camera_config import load_camera_settings
from vision.distance_config import load_distance_thresholds
from vision.model_config import load_model_settings
from vision.voice_alert import VoiceAnnouncer
from vision.yolo_thread import YoloThread


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "tmp"
# Capture/display loop interval. CPU inference can't keep up with 30 fps
# (33 ms) anyway, so polling that fast just burns CPU on resize/color
# convert/redraw for frames the AI thread throws away. ~22 fps is still
# smooth for a monitoring app and noticeably lighter on the UI thread.
FRAME_INTERVAL_MS = 45

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
QFrame#alertCard { min-width: 390px; }
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
QSlider::groove:horizontal {
    height: 6px;
    background: #334155;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f8fafc;
    border: 2px solid #2563eb;
    width: 16px;
    margin: -6px 0;
    border-radius: 9px;
}
QTextEdit {
    background: #020617;
    color: #facc15;
    border: 1px solid #243244;
    border-radius: 14px;
    padding: 12px;
    font-size: 15px;
    font-family: 'Leelawadee UI';
}
QPushButton:focus { border: 2px solid #93c5fd; }
QLabel#sectionTitle { color:#f8fafc; font-size:16px; font-weight:700; }
QLabel#sectionHint { color:#94a3b8; font-size:12px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye")
        self.resize(1360, 820)
        self.setMinimumSize(1180, 720)
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
        self.model_win = None
        self.alert_win = None
        self.test_frame = None
        self.test_mode = False
        self.test_queue = []
        self.test_index = 0
        self.test_results = []
        self.test_view_index = -1
        self.video_is_file = False
        self.video_total_frames = 0
        self.video_fps = 0.0
        self.video_slider_dragging = False
        self.danger_alarm = DangerAlarm()
        self.voice_announcer = VoiceAnnouncer()
        alert_settings = load_alert_settings()
        self.danger_alarm.set_enabled(alert_settings["siren_enabled"])
        self.voice_announcer.set_voice_model(alert_settings["voice_model"])
        self.voice_announcer.set_enabled(alert_settings["voice_enabled"])

        self._build_ui()
        self._apply_ui_labels()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_distance_status()
        self._refresh_mute_button()

        model_relative, model_path = load_model_settings()
        try:
            self.yolo = YoloThread(model_path=model_path)
            self.yolo.result_ready.connect(self.on_yolo_result)
            self.yolo.error_ready.connect(self.on_yolo_error)
            self.yolo.start()
            QTimer.singleShot(1200, self.voice_announcer.preload)
            self.set_status("Loading detection model...", "#f59e0b")
            self.set_status("พร้อมใช้งาน", "#22c55e")
        except Exception as error:
            self.alert.setText(f"Cannot load YOLO model ({model_relative}): {error}")
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
        self.btn_test = QPushButton("Test หลายภาพ")
        self.btn_test.setObjectName("secondaryButton")
        self.btn_save_results = QPushButton("บันทึกผล")
        self.btn_save_results.setObjectName("secondaryButton")
        self.btn_save_results.setEnabled(False)
        self.btn_mute = QPushButton("🔊 เสียงเตือน")
        self.btn_mute.setObjectName("secondaryButton")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setToolTip(
            "ปิด/เปิดเสียงไซเรนและเสียงพูดแจ้งเตือนเมื่ออยู่ในระยะอันตราย"
        )
        self.btn_settings = QPushButton("ตั้งค่า")
        self.btn_settings.setObjectName("settingButton")
        self.btn_settings.setToolTip("การตั้งค่า")
        self.btn_settings.setFont(QFont("Arial", 15, QFont.Bold))

        def add_group(title, buttons):
            group = QFrame(objectName="toolbarGroup")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(8, 4, 8, 4)
            group_layout.setSpacing(2)
            group_label = QLabel(title)
            group_label.setObjectName("toolbarLabel")
            group_layout.addWidget(group_label)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            for button in buttons:
                button.setFixedHeight(36)
                button.setMinimumWidth(108)
                row.addWidget(button)
            group_layout.addLayout(row)
            toolbar_layout.addWidget(group)

        add_group("แหล่งภาพ", (self.btn_open, self.btn_close, self.btn_video))
        add_group("ทดสอบและผลลัพธ์", (self.btn_test, self.btn_save_results))
        add_group("เสียง", (self.btn_mute,))

        # Restore the original flat toolbar layout.
        for group in toolbar.findChildren(QFrame, "toolbarGroup"):
            group.hide()
        for button in (self.btn_open, self.btn_close, self.btn_video, self.btn_test):
            button.setFixedSize(142, 42)
            toolbar_layout.addWidget(button)
        self.btn_save_results.setFixedSize(130, 42)
        toolbar_layout.addWidget(self.btn_save_results)
        self.btn_mute.setFixedSize(130, 42)
        toolbar_layout.addWidget(self.btn_mute)
        toolbar_layout.addStretch()
        self.btn_settings.setFixedSize(120, 42)
        toolbar_layout.addWidget(self.btn_settings)

        self.btn_open.clicked.connect(self.open_camera)
        self.btn_close.clicked.connect(lambda: self.close_camera())
        self.btn_video.clicked.connect(self.open_video)
        self.btn_test.clicked.connect(self.open_test_image)
        self.btn_save_results.clicked.connect(self.save_test_results)
        self.btn_mute.toggled.connect(self.toggle_alarm_mute)
        settings_menu = QMenu(self)
        distance_action = settings_menu.addAction("ตั้งค่าระยะ")
        zone_action = settings_menu.addAction("ตั้งค่าโซน")
        camera_action = settings_menu.addAction("ตั้งค่ากล้อง")
        model_action = settings_menu.addAction("ตั้งค่าโมเดล")
        alert_action = settings_menu.addAction("ตั้งค่าเสียงแจ้งเตือน")
        distance_action.triggered.connect(self.open_distance_setting)
        zone_action.triggered.connect(self.open_zone_setting)
        camera_action.triggered.connect(self.open_camera_setting)
        model_action.triggered.connect(self.open_model_setting)
        alert_action.triggered.connect(self.open_alert_setting)
        self.btn_settings.setMenu(settings_menu)

        content = QFrame(objectName="content")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(24, 16, 24, 24)
        content_layout.setSpacing(16)

        video_card = QFrame(objectName="videoCard")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(10)

        video_header = QHBoxLayout()
        video_title = QLabel("ภาพตรวจจับ")
        video_title.setObjectName("sectionTitle")
        video_hint = QLabel("เรียลไทม์ • ลากแถบด้านล่างเพื่อเลือกช่วงวิดีโอ")
        video_hint.setObjectName("sectionHint")
        video_header.addWidget(video_title)
        video_header.addStretch()
        video_header.addWidget(video_hint)
        video_layout.addLayout(video_header)

        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setMinimumSize(480, 380)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet(
            "background:#020617; border:1px solid #334155; border-radius:14px; color:#64748b;"
        )
        self.video.setText("ยังไม่ได้เปิดกล้องหรือวิดีโอ")
        self.video.wheelEvent = self._on_test_wheel
        video_layout.addWidget(self.video)

        self.test_navigation_widget = QWidget()
        test_navigation = QHBoxLayout(self.test_navigation_widget)
        test_navigation.setContentsMargins(0, 0, 0, 0)
        test_navigation.setSpacing(10)
        self.btn_previous_image = QPushButton("◀ ภาพก่อนหน้า")
        self.btn_previous_image.setObjectName("secondaryButton")
        self.btn_next_image = QPushButton("ภาพถัดไป ▶")
        self.btn_next_image.setObjectName("secondaryButton")
        self.test_image_counter = QLabel("ยังไม่มีผลภาพ")
        self.test_image_counter.setAlignment(Qt.AlignCenter)
        self.test_image_counter.setStyleSheet(
            "color:#94a3b8; font-size:13px; font-weight:700;"
        )
        self.btn_previous_image.setEnabled(False)
        self.btn_next_image.setEnabled(False)
        self.btn_previous_image.clicked.connect(self.show_previous_test_result)
        self.btn_next_image.clicked.connect(self.show_next_test_result)
        test_navigation.addWidget(self.btn_previous_image)
        test_navigation.addStretch()
        test_navigation.addWidget(self.test_image_counter)
        test_navigation.addStretch()
        test_navigation.addWidget(self.btn_next_image)
        self.test_navigation_widget.setVisible(False)
        video_layout.addWidget(self.test_navigation_widget)

        self.video_timeline_widget = QWidget()
        video_timeline = QHBoxLayout(self.video_timeline_widget)
        video_timeline.setContentsMargins(0, 0, 0, 0)
        video_timeline.setSpacing(10)
        self.video_time_label = QLabel("00:00 / 00:00")
        self.video_time_label.setMinimumWidth(105)
        self.video_time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.video_time_label.setStyleSheet(
            "color:#cbd5e1; font-size:12px; font-weight:700;"
        )
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setRange(0, 0)
        self.video_slider.sliderPressed.connect(self._on_video_slider_pressed)
        self.video_slider.sliderReleased.connect(self._on_video_slider_released)
        self.video_slider.valueChanged.connect(self._preview_video_slider_time)
        video_timeline.addWidget(self.video_slider)
        video_timeline.addWidget(self.video_time_label)
        self.video_timeline_widget.setVisible(False)
        video_layout.addWidget(self.video_timeline_widget)

        alert_card = QFrame(objectName="alertCard")
        alert_card.setMinimumWidth(390)
        alert_card.setMaximumWidth(430)
        alert_layout = QVBoxLayout(alert_card)
        alert_layout.setContentsMargins(14, 14, 14, 14)
        alert_layout.setSpacing(10)

        alert_title = QLabel("รายการแจ้งเตือน")
        alert_title.setFont(QFont("Leelawadee UI", 17, QFont.Bold))
        alert_hint = QLabel("แสดงวัตถุที่อยู่ในระยะอันตรายและระวัง")
        alert_hint.setObjectName("sectionHint")
        self.alert = QTextEdit()
        self.alert.setReadOnly(True)
        self.alert.setHtml(
            "<span style='color:#94a3b8;'>เปิดกล้องหรืออัปโหลดวิดีโอเพื่อเริ่มตรวจจับ</span>"
        )
        alert_layout.addWidget(alert_title)
        alert_layout.addWidget(alert_hint)

        self.source_metric = self._make_metric("แหล่งภาพ", self.source_name)
        self.fps_metric = self._make_metric("อัตราเฟรม / AI", "--")
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

    def _apply_ui_labels(self):
        """Normalize the primary controls to concise, readable Thai labels."""
        labels = {
            self.btn_open: "เปิดกล้อง",
            self.btn_close: "ปิดกล้อง",
            self.btn_video: "เปิดวิดีโอ",
            self.btn_test: "ทดสอบรูปภาพ",
            self.btn_save_results: "บันทึกผล",
            self.btn_settings: "⚙ ตั้งค่า",
        }
        for widget, text in labels.items():
            widget.setText(text)
        self.btn_mute.setText("🔊 เสียงแจ้งเตือน")
        self.btn_mute.setToolTip("เปิดหรือปิดเสียงไซเรนและเสียงพูดแจ้งเตือน")
        self.btn_settings.setToolTip("ตั้งค่าระยะ โซน กล้อง โมเดล และเสียง")
        actions = self.btn_settings.menu().actions()
        menu_labels = [
            "ตั้งค่าระยะเตือน",
            "ตั้งค่าโซน",
            "ตั้งค่ากล้อง",
            "ตั้งค่าโมเดล",
            "ตั้งค่าเสียงแจ้งเตือน",
        ]
        for action, text in zip(actions, menu_labels):
            action.setText(text)
        self.btn_previous_image.setText("← ก่อนหน้า")
        self.btn_next_image.setText("ถัดไป →")
        self.video.setText("ยังไม่ได้เลือกแหล่งภาพ\nเปิดกล้อง วิดีโอ หรือทดสอบรูปภาพ")
        self.test_image_counter.setText("ยังไม่มีผลทดสอบ")
        self.status_badge.setText("กำลังเริ่มระบบ")
        self.alert.setHtml(
            "<span style='color:#94a3b8;'>เปิดแหล่งภาพเพื่อเริ่มตรวจจับวัตถุ</span>"
        )

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

    def toggle_alarm_mute(self, muted):
        self.danger_alarm.set_muted(muted)
        self.voice_announcer.set_muted(muted)
        self._refresh_mute_button()

    def _refresh_mute_button(self):
        """Reflect hardware availability, the alert-settings on/off state,
        and the mute toggle itself on the toolbar button."""
        unavailable = not self.danger_alarm.available and not self.voice_announcer.available
        disabled_in_settings = not self.danger_alarm.enabled and not self.voice_announcer.enabled

        if unavailable or disabled_in_settings:
            self.btn_mute.setChecked(True)
            self.btn_mute.setEnabled(False)
            self.btn_mute.setText(
                "🔇 ไม่มีเสียง" if unavailable else "🔇 ปิดไว้ในตั้งค่า"
            )
            return

        self.btn_mute.setEnabled(True)
        self.btn_mute.setText("🔇 ปิดเสียง" if self.btn_mute.isChecked() else "🔊 เสียงเตือน")

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
        self.video_is_file = False
        self.test_navigation_widget.setVisible(False)
        self.video_timeline_widget.setVisible(False)
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
        self.timer.start(FRAME_INTERVAL_MS)
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
        self.video_is_file = True
        self.video_total_frames = max(
            0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.video_slider.setRange(0, max(0, self.video_total_frames - 1))
        self.video_slider.setValue(0)
        self.video_time_label.setText(
            f"00:00 / "
            f"{self._format_video_time(self.video_total_frames / self.video_fps)}"
        )
        self.test_navigation_widget.setVisible(False)
        self.video_timeline_widget.setVisible(True)
        self.last_time = time.time()
        self.source_name = Path(path).name
        self.source_metric.value_label.setText(self.source_name)
        self.timer.start(FRAME_INTERVAL_MS)
        self.alert.setText("เปิดวิดีโอแล้ว")
        self.set_status("กำลังทำงาน", "#16a34a")

    @staticmethod
    def _format_video_time(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _on_video_slider_pressed(self):
        if not self.video_is_file:
            return
        self.video_slider_dragging = True
        self.timer.stop()

    def _preview_video_slider_time(self, frame_number):
        if not self.video_is_file or self.video_fps <= 0:
            return
        current = self._format_video_time(frame_number / self.video_fps)
        duration = self._format_video_time(
            self.video_total_frames / self.video_fps
        )
        self.video_time_label.setText(f"{current} / {duration}")

    def _on_video_slider_released(self):
        if not self.video_is_file or self.cap is None:
            return
        frame_number = self.video_slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if self.yolo is not None:
            self.yolo.clear_frame()
        self.video_slider_dragging = False
        self.last_time = time.time()
        self.update_frame()
        if self.cap is not None:
            self.timer.start(FRAME_INTERVAL_MS)

    def open_test_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "เลือกภาพสำหรับ Test (เลือกได้หลายภาพ)",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if not paths:
            return

        self.close_camera(clear_display=False)
        self.test_mode = True
        self.test_queue = list(paths)
        self.test_index = 0
        self.test_results = []
        self.test_view_index = -1
        self.btn_save_results.setEnabled(False)
        self.test_navigation_widget.setVisible(True)
        self.video_timeline_widget.setVisible(False)
        self._update_test_navigation()

        if self.yolo is None:
            self.alert.setText("โมเดลยังไม่พร้อมใช้งาน")
            self.set_status("โมเดลไม่พร้อม", "#ef4444")
            return
        self._load_next_test_image()

    def _load_next_test_image(self):
        if not self.test_mode:
            return
        while self.test_mode and self.test_index < len(self.test_queue):
            path = self.test_queue[self.test_index]
            image = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                self.test_results.append(
                    {
                        "path": path,
                        "detections": [],
                        "image": None,
                        "error": True,
                        "info": (
                            "<span style='color:#ef4444;'>"
                            "ไม่สามารถเปิดไฟล์ภาพนี้ได้</span>"
                        ),
                    }
                )
                self.test_index += 1
                continue

            self.test_frame = cv2.resize(image, (640, 640))
            self.current_frame = self.test_frame.copy()
            self.source_name = Path(path).name
            self.source_metric.value_label.setText(
                f"{self.source_name} ({self.test_index + 1}/{len(self.test_queue)})"
            )
            self.fps_metric.value_label.setText("กำลังตรวจจับ...")
            self.last_info = (
                f"<span style='color:#94a3b8;'>กำลังตรวจภาพ "
                f"{self.test_index + 1} จาก {len(self.test_queue)} "
                "โดยไม่ใช้โซน...</span>"
            )
            self.alert.setHtml(self.last_info)
            self._display_frame(
                self.test_frame.copy(),
                draw_zone=False,
                show_detections=False,
            )
            self.set_status("กำลัง Test หลายภาพ", "#2563eb")
            self._submit_test_frame()
            return

        if self.test_mode:
            self._finish_test_batch()

    def _submit_test_frame(self):
        if self.test_mode and self.test_frame is not None and self.yolo is not None:
            if not self.yolo.update_frame(self.test_frame, use_zone=False):
                QTimer.singleShot(50, self._submit_test_frame)

    def _finish_test_batch(self):
        self.test_mode = False
        self.test_frame = None
        self.btn_save_results.setEnabled(bool(self.test_results))
        detected = sum(len(item["detections"]) for item in self.test_results)
        failed = sum(bool(item.get("error")) for item in self.test_results)
        self.alert.setHtml(
            f"<span style='color:#22c55e;'>Test ครบ {len(self.test_results)} ภาพ "
            f"พบวัตถุรวม {detected} รายการ</span>"
            + (
                f"<br><span style='color:#ef4444;'>เปิดภาพไม่สำเร็จ {failed} ภาพ</span>"
                if failed
                else ""
            )
        )
        self.fps_metric.value_label.setText("เสร็จแล้ว")
        self.set_status("Test เสร็จแล้ว", "#22c55e")
        if self.test_results:
            self.show_test_result(len(self.test_results) - 1)
        else:
            self._update_test_navigation()

    def show_test_result(self, index):
        if not self.test_results:
            return
        self.test_view_index = max(0, min(index, len(self.test_results) - 1))
        item = self.test_results[self.test_view_index]
        self.source_name = Path(item["path"]).name
        self.source_metric.value_label.setText(self.source_name)
        if item["image"] is not None:
            self._display_frame(
                item["image"].copy(),
                draw_zone=False,
                show_detections=False,
            )
        else:
            self.video.clear()
            self.video.setText("ไม่สามารถแสดงภาพนี้ได้")
        self.alert.setHtml(
            item.get(
                "info",
                "<span style='color:#ef4444;'>ไม่สามารถประมวลผลภาพนี้ได้</span>",
            )
        )
        self._update_test_navigation()

    def show_previous_test_result(self):
        if self.test_results:
            self.show_test_result(self.test_view_index - 1)

    def show_next_test_result(self):
        if self.test_results:
            self.show_test_result(self.test_view_index + 1)

    def _on_test_wheel(self, event):
        if not self.test_results or self.test_mode:
            event.ignore()
            return
        if event.angleDelta().y() > 0:
            self.show_previous_test_result()
        elif event.angleDelta().y() < 0:
            self.show_next_test_result()
        event.accept()

    def _update_test_navigation(self):
        total = len(self.test_results)
        if total == 0:
            self.test_image_counter.setText("ยังไม่มีผลภาพ")
            self.btn_previous_image.setEnabled(False)
            self.btn_next_image.setEnabled(False)
            return
        current = max(0, self.test_view_index)
        self.test_image_counter.setText(f"ภาพ {current + 1} / {total}")
        navigation_enabled = not self.test_mode
        self.btn_previous_image.setEnabled(navigation_enabled and current > 0)
        self.btn_next_image.setEnabled(
            navigation_enabled and current < total - 1
        )

    def save_test_results(self):
        if not self.test_results:
            return
        output_dir = QFileDialog.getExistingDirectory(
            self, "เลือกโฟลเดอร์สำหรับบันทึกผล"
        )
        if not output_dir:
            return

        output_path = Path(output_dir)
        csv_path = output_path / "detection_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "image",
                    "object_no",
                    "label",
                    "distance_m",
                    "status",
                    "confidence",
                    "box",
                    "error",
                ]
            )
            for result_index, item in enumerate(self.test_results, 1):
                image_name = Path(item["path"]).name
                detections = item["detections"]
                if item["image"] is not None:
                    result_name = (
                        f"{result_index:03d}_{Path(image_name).stem}_detected.jpg"
                    )
                    encoded_ok, encoded = cv2.imencode(".jpg", item["image"])
                    if encoded_ok:
                        encoded.tofile(str(output_path / result_name))
                if not detections:
                    writer.writerow(
                        [
                            image_name,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "read/inference error" if item.get("error") else "",
                        ]
                    )
                    continue
                for index, detection in enumerate(detections, 1):
                    writer.writerow(
                        [
                            image_name,
                            index,
                            detection["label"],
                            detection["dist"],
                            detection["status"],
                            detection["score"],
                            detection["box"],
                            "",
                        ]
                    )

        self.alert.setHtml(
            f"<span style='color:#22c55e;'>บันทึกภาพผลลัพธ์และไฟล์ "
            f"detection_results.csv ที่ {output_path}</span>"
        )

    def close_camera(self, clear_display=True):
        self.timer.stop()
        self.danger_alarm.stop()
        self.voice_announcer.stop()
        self.test_mode = False
        self.test_frame = None
        self.test_queue = []
        self.test_index = 0
        self.test_results = []
        self.test_view_index = -1
        self.btn_save_results.setEnabled(False)
        self._update_test_navigation()
        self.test_navigation_widget.setVisible(False)
        self.video_timeline_widget.setVisible(False)
        self.video_is_file = False
        self.video_total_frames = 0
        self.video_fps = 0.0
        self.video_slider_dragging = False
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
            if self.video_is_file:
                self.timer.stop()
                self.video_slider.setValue(
                    max(0, self.video_total_frames - 1)
                )
                self.alert.setText(
                    "วิดีโอจบแล้ว สามารถลากแถบเวลาเพื่อดูช่วงอื่นได้"
                )
                self.set_status("วิดีโอจบแล้ว", "#2563eb")
                return
            self.close_camera()
            self.alert.setText("ไม่สามารถอ่านเฟรมจากกล้องได้")
            return

        frame = cv2.resize(frame, (640, 640))
        self.current_frame = frame.copy()
        if self.video_is_file and not self.video_slider_dragging:
            current_frame = max(
                0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            )
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(current_frame)
            self.video_slider.blockSignals(False)
            self._preview_video_slider_time(current_frame)
        if self.yolo is not None:
            self.yolo.update_frame(frame)

        self._display_frame(frame, draw_zone=True)

    def _display_frame(self, frame, draw_zone, show_detections=True):
        detections = (
            self.yolo.get_detections()
            if show_detections and self.yolo
            else []
        )
        for detection in detections:
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
            danger_detections = [d for d in detections if d["status"] == "DANGER"]
            alert_detections = [
                d for d in detections if d["status"] in ("DANGER", "WARNING")
            ]
            nearest_danger = min(
                danger_detections, key=lambda d: d["dist"], default=None
            )
            # DANGER always takes priority over WARNING, even when a WARNING
            # object is physically a little closer.
            nearest_alert = nearest_danger or min(
                alert_detections, key=lambda d: d["dist"], default=None
            )
            self.danger_alarm.set_active(nearest_danger is not None)
            self.voice_announcer.announce(
                nearest_alert["label"] if nearest_alert else None,
                nearest_alert["dist"] if nearest_alert else None,
                nearest_alert["status"] if nearest_alert else "SAFE",
            )
        return frame

    def on_yolo_result(self, info):
        self.last_info = info
        if self.yolo is not None and self.yolo.model_ready:
            self.set_status("พร้อมใช้งาน", "#22c55e")
        if self.test_mode and self.test_frame is not None:
            annotated = self._display_frame(
                self.test_frame.copy(), draw_zone=False
            )
            detections = (
                self.yolo.get_detections(predict=False) if self.yolo is not None else []
            )
            self.test_results.append(
                {
                    "path": self.test_queue[self.test_index],
                    "detections": detections,
                    "image": annotated,
                    "error": False,
                    "info": info,
                }
            )
            self.alert.setHtml(info)
            ai_fps = self.yolo.inference_fps if self.yolo is not None else 0.0
            self.fps_metric.value_label.setText(f"AI {ai_fps:.1f} FPS")
            self.test_index += 1
            QTimer.singleShot(50, self._load_next_test_image)

    def on_yolo_error(self, message):
        self.last_info = f"<span style='color:#ef4444;'>{message}</span>"
        if self.test_mode:
            self.alert.setHtml(self.last_info)
            if self.test_index < len(self.test_queue):
                self.test_results.append(
                    {
                        "path": self.test_queue[self.test_index],
                        "detections": [],
                        "image": self.test_frame.copy()
                        if self.test_frame is not None
                        else None,
                        "error": True,
                        "info": self.last_info,
                    }
                )
                self.test_index += 1
                QTimer.singleShot(50, self._load_next_test_image)
        self.set_status("ตรวจจับผิดพลาด", "#ef4444")

    def open_distance_setting(self):
        if self.distance_win is None:
            self.distance_win = DistanceSettingWindow()
            self.distance_win.thresholds_saved.connect(self.on_thresholds_saved)
        self._show_single_settings_window(self.distance_win)

    def open_camera_setting(self):
        if self.camera_win is None:
            self.camera_win = CameraSettingWindow()
            self.camera_win.settings_saved.connect(self.on_camera_settings_saved)
        self._show_single_settings_window(self.camera_win)

    def on_camera_settings_saved(self, camera_index, focal_length):
        if self.yolo is not None:
            self.yolo.update_camera_settings(focal_length)
        self.last_info = (
            f"<span style='color:#e5e7eb;'>บันทึกค่ากล้องแล้ว: "
            f"Camera {camera_index}, Focal Length {focal_length:.1f} px</span>"
        )
        self.alert.setHtml(self.last_info)

    def open_model_setting(self):
        if self.model_win is None:
            self.model_win = ModelSettingWindow()
            self.model_win.model_selected.connect(self.switch_model)
        self._show_single_settings_window(self.model_win)

    def switch_model(self, relative_path):
        """Restart the detection thread on the newly selected model."""
        self.close_camera(clear_display=False)

        if self.yolo is not None:
            self.yolo.result_ready.disconnect(self.on_yolo_result)
            self.yolo.error_ready.disconnect(self.on_yolo_error)
            self.yolo.stop()
            self.yolo = None

        model_relative, model_path = load_model_settings()
        try:
            self.yolo = YoloThread(model_path=model_path)
            self.yolo.result_ready.connect(self.on_yolo_result)
            self.yolo.error_ready.connect(self.on_yolo_error)
            self.yolo.start()
            QTimer.singleShot(1200, self.voice_announcer.preload)
            self.last_info = (
                f"<span style='color:#22c55e;'>เปลี่ยนโมเดลเป็น \"{model_relative}\" แล้ว</span>"
            )
            self.alert.setHtml(self.last_info)
            self.set_status("พร้อมใช้งาน", "#22c55e")
        except Exception as error:
            self.alert.setText(f"Cannot load YOLO model ({model_relative}): {error}")
            self.set_status("โหลดโมเดลไม่สำเร็จ", "#ef4444")

    def open_alert_setting(self):
        if self.alert_win is None:
            self.alert_win = AlertSettingWindow()
            self.alert_win.settings_saved.connect(self.on_alert_settings_saved)
        self._show_single_settings_window(self.alert_win)

    def on_alert_settings_saved(self, siren_enabled, voice_enabled, voice_model):
        self.danger_alarm.set_enabled(siren_enabled)
        self.voice_announcer.set_voice_model(voice_model)
        self.voice_announcer.set_enabled(voice_enabled)
        self._refresh_mute_button()
        self.last_info = (
            "<span style='color:#e5e7eb;'>บันทึกการตั้งค่าเสียงแจ้งเตือนแล้ว: "
            f"ไซเรน {'เปิด' if siren_enabled else 'ปิด'}, "
            f"เสียงพูด {'เปิด' if voice_enabled else 'ปิด'}</span>"
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

        if self.zone_win is None or not self.zone_win.isVisible():
            self.zone_win = ZoneSettingWindow(str(image_path) if image_path else None)
            if self.yolo is not None:
                self.zone_win.zone_saved.connect(self.yolo.invalidate_zone)
        self._show_single_settings_window(self.zone_win)

    def _show_single_settings_window(self, window):
        """Keep settings focused: only one settings page is visible at a time."""
        settings_windows = (
            self.distance_win,
            self.zone_win,
            self.camera_win,
            self.model_win,
            self.alert_win,
        )
        for other in settings_windows:
            if other is not None and other is not window:
                other.hide()
        window.setWindowModality(Qt.ApplicationModal)
        window.show()
        window.raise_()
        window.activateWindow()

    def closeEvent(self, event):
        self.close_camera()
        self.danger_alarm.stop()
        self.voice_announcer.stop()
        if self.yolo is not None and self.yolo.isRunning():
            if not self.yolo.stop():
                self.alert.setText("กำลังรอให้ระบบตรวจจับหยุดทำงาน...")
                event.ignore()
                QTimer.singleShot(250, self.close)
                return
        event.accept()
