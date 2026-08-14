import csv
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.settings_window import SettingsWindow
from ui.icons import icon
from vision.alert_config import load_alert_settings
from vision.alert_sound import DangerAlarm
from vision.camera_config import load_camera_settings
from vision.capture_thread import CaptureThread
from vision.distance_config import load_distance_thresholds
from vision.model_config import load_model_settings
from vision.voice_alert import VoiceAnnouncer
from vision.yolo_thread import YoloThread


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "tmp"
# Capture/display loop interval. The display uses the source's latest frame;
# the detector is independently throttled in YoloThread. Keeping the UI at
# up to 30 FPS avoids visible judder without queueing old frames.
FRAME_INTERVAL_MS = 33
# Keep the full aspect ratio while limiting the copy/colour-conversion work
# sent to Qt. The QLabel scales this to its available area. The model still
# receives its own exact 640x640 letterbox.
DISPLAY_FRAME_SIZE = (640, 360)
MODEL_FRAME_SIZE = (640, 640)
UI_STATUS_UPDATE_INTERVAL_S = 0.20


def _video_timer_interval(video_fps):
    """Return a UI-friendly display interval without oversampling the UI."""
    try:
        fps = float(video_fps)
    except (TypeError, ValueError):
        fps = 0.0
    if not np.isfinite(fps) or fps <= 0:
        return FRAME_INTERVAL_MS
    return max(FRAME_INTERVAL_MS, int(round(1000.0 / min(fps, 30.0))))

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
QFrame#bottomStatus {
    background:#111C2E;
    border-top:1px solid #26364D;
}
QFrame#videoCard, QFrame#alertCard {
    background: #121d2f;
    border: 1px solid #2b3a52;
    border-radius: 16px;
}
QFrame#alertCard { min-width: 350px; }
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
    background: #111C2E;
    color: #facc15;
    border: 1px solid #243244;
    border-radius: 10px;
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

        self.capture = None
        self._last_capture_frame_number = -1
        self.current_frame = None
        self.last_time = time.time()
        self.fps = 0.0
        self.source_name = "No source"
        self.last_info = ""
        self.yolo = None
        self.settings_win = None
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
        self._last_ui_status_update = 0.0
        self._last_detection_counts = None
        self._last_video_time_text = None
        self._last_alert_html = None
        self.model_name = "กำลังโหลด"
        self._stopping_yolo = None
        self._pending_model = None
        self._stopping_capture = None
        self._pending_video_path = None
        self.danger_alarm = DangerAlarm()
        self.voice_announcer = VoiceAnnouncer()
        alert_settings = load_alert_settings()
        self.danger_alarm.set_enabled(alert_settings["siren_enabled"])
        self.danger_alarm.set_volume(alert_settings["siren_volume"])
        self.voice_announcer.set_voice_model(alert_settings["voice_model"])
        self.voice_announcer.set_enabled(alert_settings["voice_enabled"])
        self.voice_announcer.set_volume(alert_settings["voice_volume"])

        self._build_ui()
        self._apply_ui_labels()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)
        self.update_distance_status()
        self._refresh_mute_button()

        model_relative, model_path = load_model_settings()
        self.model_name = model_relative
        try:
            self.yolo = YoloThread(model_path=model_path)
            self.yolo.result_ready.connect(self.on_yolo_result)
            self.yolo.error_ready.connect(self.on_yolo_error)
            self.yolo.model_ready_signal.connect(self.on_yolo_ready)
            # Keep inference below the UI/capture threads on CPU-only systems.
            # A slightly slower detector is preferable to a frozen preview.
            # Prioritize the preview pipeline; YOLO can process the newest
            # frame later without making the video appear frozen.
            self.yolo.start(QThread.LowestPriority)
            self.set_status("กำลังเตรียมโมเดล", "#f59e0b")
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
        subtitle.setText("ตรวจจับวัตถุ ประเมินระยะ และแจ้งเตือนความเสี่ยง")

        self.status_badge = QLabel("กำลังเริ่มระบบ")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedWidth(92)
        self.status_badge.setStyleSheet(
            "background:#162238; color:#CBD5E1; border:1px solid #334155; border-radius:8px; padding:6px 10px; font-size:12px; font-weight:700;"
        )

        self.camera_badge = QLabel("กล้อง ปิด")
        self.camera_badge.setStyleSheet(
            "background:#162238; color:#94A3B8; border:1px solid #26364D; "
            "border-radius:8px; padding:6px 10px; font-size:12px; font-weight:700;"
        )

        header_layout.addLayout(title_box)
        header_layout.addStretch()
        header_layout.addWidget(self.camera_badge)
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
            "ปิด/เปิดเสียงไซเรนและเสียงพูดแจ้งเตือนในระยะระวัง/อันตราย"
        )
        self.btn_settings = QPushButton("ตั้งค่า")
        self.btn_settings.setObjectName("settingButton")
        self.btn_settings.setToolTip("การตั้งค่า")
        self.btn_settings.setFont(QFont("Arial", 15, QFont.Bold))
        self.btn_open.setIcon(icon("camera"))
        self.btn_video.setIcon(icon("video"))
        self.btn_test.setIcon(icon("image"))
        self.btn_close.setIcon(icon("stop"))
        self.btn_save_results.setIcon(icon("save"))
        self.btn_mute.setIcon(icon("volume-x" if self.btn_mute.isChecked() else "volume"))
        self.btn_settings.setIcon(icon("settings"))

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
        # clicked(bool) passes the checked state; open_video's optional path
        # argument is reserved for queued programmatic switches.
        self.btn_video.clicked.connect(lambda _checked=False: self.open_video())
        self.btn_test.clicked.connect(self.open_test_image)
        self.btn_save_results.clicked.connect(self.save_test_results)
        self.btn_mute.toggled.connect(self.toggle_alarm_mute)
        self.btn_settings.clicked.connect(self.open_settings)

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
        alert_card.setMinimumWidth(350)
        alert_card.setMaximumWidth(390)
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

        count_row = QHBoxLayout()
        count_row.setSpacing(8)
        self.danger_count = self._make_count("อันตราย", "0", "#EF4444")
        self.warning_count = self._make_count("ระวัง", "0", "#F59E0B")
        count_row.addWidget(self.danger_count)
        self.safe_count = self._make_count("ปลอดภัย", "0", "#22C55E")
        count_row.addWidget(self.warning_count)
        count_row.addWidget(self.safe_count)
        alert_layout.addLayout(count_row)

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
        bottom_status = QFrame(objectName="bottomStatus")
        bottom_layout = QHBoxLayout(bottom_status)
        bottom_layout.setContentsMargins(24, 6, 24, 6)
        self.bottom_status = QLabel("แหล่งภาพ: ไม่มี | โมเดล: กำลังโหลด | FPS: -- | โซน: เปิด | เสียง: เปิด")
        self.bottom_status.setStyleSheet("color:#94A3B8; font-size:11px;")
        bottom_layout.addWidget(self.bottom_status)
        bottom_layout.addStretch()
        main.addWidget(bottom_status)

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
        self.btn_mute.setToolTip(
            "เปิดหรือปิดเสียงไซเรนและเสียงพูดแจ้งเตือนในระยะระวัง/อันตราย"
        )
        self.btn_settings.setToolTip("ตั้งค่าระยะ โซน กล้อง โมเดล และเสียง")
        actions = []
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

        # Final visible labels are kept here so the primary shell remains
        # readable even when older strings in the legacy handlers are kept.
        for widget, text in (
            (self.btn_open, "เปิดกล้อง"),
            (self.btn_close, "หยุด"),
            (self.btn_video, "เปิดวิดีโอ"),
            (self.btn_test, "เปิดรูปภาพ"),
            (self.btn_save_results, "บันทึกผล"),
            (self.btn_mute, "เสียงแจ้งเตือน"),
            (self.btn_settings, "ตั้งค่า"),
        ):
            widget.setText(text)
        self.video.setText("ยังไม่ได้เลือกแหล่งภาพ\nเปิดกล้อง วิดีโอ หรือรูปภาพเพื่อเริ่มตรวจจับ")
        self.test_image_counter.setText("ยังไม่มีผลทดสอบ")
        self.alert.setHtml("<span style='color:#94a3b8;'>ยังไม่มีการแจ้งเตือน</span>")

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

    @staticmethod
    def _make_count(title, value, color):
        label = QLabel(f"{title}: {value}")
        label.setStyleSheet(
            f"background:#111C2E; color:{color}; border-left:3px solid {color}; "
            "border-radius:6px; padding:7px 9px; font-size:12px; font-weight:700;"
        )
        label.value = value
        label.title = title
        return label

    def toggle_alarm_mute(self, muted):
        self.danger_alarm.set_muted(muted)
        self.voice_announcer.set_muted(muted)
        self._refresh_mute_button()

    def _refresh_mute_button(self):
        """Reflect hardware availability, the alert-settings on/off state,
        and the mute toggle itself on the toolbar button."""
        QTimer.singleShot(0, self._normalize_mute_label)
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

    def _normalize_mute_label(self):
        """Keep the SVG icon and a short text label; never append emoji."""
        unavailable = not self.danger_alarm.available and not self.voice_announcer.available
        disabled_in_settings = not self.danger_alarm.enabled and not self.voice_announcer.enabled
        if unavailable:
            self.btn_mute.setText("ไม่มีเสียง")
        elif disabled_in_settings:
            self.btn_mute.setText("ปิดในการตั้งค่า")
        else:
            self.btn_mute.setText("เปิดเสียง" if self.btn_mute.isChecked() else "ปิดเสียง")
        self.btn_mute.setIcon(icon("volume-x" if self.btn_mute.isChecked() else "volume"))

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
        if self.capture is not None:
            if self.capture.isRunning() or self._stopping_capture is not None:
                return
            self.capture = None
        if self._stopping_capture is not None:
            return
        if self.yolo is None or not self.yolo.isRunning():
            self.alert.setText("โมเดลยังไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์โมเดลแล้วเปิดโปรแกรมใหม่")
            self.set_status("โมเดลไม่พร้อมใช้งาน", "#ef4444")
            return
        self.test_mode = False
        self.test_frame = None
        self.video_is_file = False
        self.test_navigation_widget.setVisible(False)
        self.video_timeline_widget.setVisible(False)
        camera_index = load_camera_settings()["camera_index"]
        self.capture = CaptureThread(
            camera_index,
            is_video=False,
            output_size=DISPLAY_FRAME_SIZE,
            model_size=MODEL_FRAME_SIZE,
            parent=self,
        )
        self.capture.error_ready.connect(self._on_capture_error)
        self.capture.end_of_stream.connect(self._on_capture_end)
        # Normal priority prevents a decoder burst from starving the GUI
        # event loop. Capture remains on its own thread.
        self.capture.start(QThread.NormalPriority)
        self.yolo.set_frame_source(self.capture.get_latest_model)
        self.voice_announcer.set_live_source(True)
        self._last_capture_frame_number = -1
        self.last_time = time.time()
        self.source_name = f"Camera {camera_index}"
        self.camera_badge.setText("กล้อง เปิด")
        self.camera_badge.setStyleSheet("background:#123524; color:#86EFAC; border:1px solid #22C55E; border-radius:8px; padding:6px 10px; font-size:12px; font-weight:700;")
        self.source_metric.value_label.setText(self.source_name)
        self.timer.start(FRAME_INTERVAL_MS)
        self._reset_display_throttle()
        self.last_info = "<span style='color:#94a3b8;'>เปิดกล้องแล้ว</span>"
        self._last_alert_html = None
        self.alert.setHtml(self.last_info)
        if self.yolo is not None and not self.yolo.model_ready:
            self.set_status("กำลังเตรียมโมเดล", "#f59e0b")
        else:
            self.set_status("กำลังทำงาน", "#16a34a")

    def open_video(self, path=None, _after_stop=False):
        if self.yolo is None or not self.yolo.isRunning():
            self.alert.setText("โมเดลยังไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์โมเดลแล้วเปิดโปรแกรมใหม่")
            self.set_status("โมเดลไม่พร้อมใช้งาน", "#ef4444")
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
        if not path:
            return

        if not _after_stop:
            self._pending_video_path = path
        if not _after_stop and not self.close_camera(
            clear_display=False, preserve_pending=True
        ):
            self.alert.setText("กำลังปิดแหล่งภาพเดิม กรุณาลองใหม่อีกครั้ง")
            self.set_status("กำลังปิดแหล่งภาพ", "#f59e0b")
            return
        self._pending_video_path = None
        self.test_mode = False
        self.test_frame = None
        metadata_capture = cv2.VideoCapture(path)
        if not metadata_capture.isOpened():
            metadata_capture.release()
            self.alert.setText("ไม่สามารถเปิดวิดีโอได้")
            self.set_status("วิดีโอผิดพลาด", "#ef4444")
            return
        self.video_is_file = True
        self.video_total_frames = max(
            0, int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.video_fps = metadata_capture.get(cv2.CAP_PROP_FPS)
        metadata_capture.release()
        if not np.isfinite(self.video_fps) or self.video_fps <= 0:
            self.video_fps = 30.0
        self.video_slider.setRange(0, max(0, self.video_total_frames - 1))
        self.video_slider.setValue(0)
        self.video_time_label.setText(
            f"00:00 / "
            f"{self._format_video_time(self.video_total_frames / self.video_fps)}"
        )
        self.test_navigation_widget.setVisible(False)
        self.video_timeline_widget.setVisible(True)
        self.capture = CaptureThread(
            path,
            is_video=True,
            video_fps=self.video_fps,
            output_size=DISPLAY_FRAME_SIZE,
            model_size=MODEL_FRAME_SIZE,
            parent=self,
        )
        self.capture.error_ready.connect(self._on_capture_error)
        self.capture.end_of_stream.connect(self._on_capture_end)
        self.capture.start(QThread.NormalPriority)
        self.yolo.set_frame_source(self.capture.get_latest_model)
        self.voice_announcer.set_live_source(True)
        self._last_capture_frame_number = -1
        self.last_time = time.time()
        self.source_name = Path(path).name
        self.source_metric.value_label.setText(self.source_name)
        self.timer.start(_video_timer_interval(self.video_fps))
        self._reset_display_throttle()
        self.last_info = "<span style='color:#94a3b8;'>เปิดวิดีโอแล้ว</span>"
        self._last_alert_html = None
        self.alert.setHtml(self.last_info)
        if self.yolo is not None and not self.yolo.model_ready:
            self.set_status("กำลังเตรียมโมเดล", "#f59e0b")
        else:
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
        if not self.video_is_file or self.capture is None:
            return
        self.video_slider_dragging = True
        self.capture.pause()
        self.timer.stop()

    def _preview_video_slider_time(self, frame_number):
        if not self.video_is_file or self.video_fps <= 0:
            return
        current = self._format_video_time(frame_number / self.video_fps)
        duration = self._format_video_time(
            self.video_total_frames / self.video_fps
        )
        text = f"{current} / {duration}"
        if text != self._last_video_time_text:
            self.video_time_label.setText(text)
            self._last_video_time_text = text

    def _reset_display_throttle(self):
        self._last_ui_status_update = 0.0
        self._last_detection_counts = None
        self._last_video_time_text = None

    def _on_video_slider_released(self):
        if not self.video_is_file or self.capture is None:
            return
        frame_number = self.video_slider.value()
        self.capture.seek(frame_number)
        self.capture.resume()
        if self.yolo is not None:
            self.yolo.clear_frame()
        self.video_slider_dragging = False
        self._last_capture_frame_number = -1
        self.last_time = time.time()
        self.timer.start(_video_timer_interval(self.video_fps))

    def open_test_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "เลือกภาพสำหรับ Test (เลือกได้หลายภาพ)",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if not paths:
            return

        if not self.close_camera(clear_display=False):
            self.alert.setText("กำลังปิดแหล่งภาพเดิม กรุณาลองใหม่อีกครั้ง")
            self.set_status("กำลังปิดแหล่งภาพ", "#f59e0b")
            return
        if self.yolo is None or not self.yolo.isRunning():
            self.alert.setText("โมเดลยังไม่พร้อมใช้งาน")
            self.set_status("โมเดลไม่พร้อม", "#ef4444")
            return
        self.test_mode = True
        self.test_queue = list(paths)
        self.test_index = 0
        self.test_results = []
        self.test_view_index = -1
        self.btn_save_results.setEnabled(False)
        self.test_navigation_widget.setVisible(True)
        self.video_timeline_widget.setVisible(False)
        self._update_test_navigation()
        self._load_next_test_image()

    @staticmethod
    def _encode_result_image(image):
        """Store batch results compactly instead of retaining raw arrays."""
        if image is None:
            return None
        encoded_ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        return encoded.tobytes() if encoded_ok else None

    @staticmethod
    def _decode_result_image(data):
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        try:
            return cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except (TypeError, ValueError):
            return None

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

            # Keep the original aspect ratio for the preview. YoloThread
            # letterboxes this frame internally before 640x640 inference.
            self.test_frame = image
            self.current_frame = self.test_frame
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
        if not (self.test_mode and self.test_frame is not None):
            return
        if self.yolo is None or not self.yolo.isRunning():
            self._abort_test_batch("โมเดลหยุดทำงานก่อนตรวจภาพเสร็จ")
            return
        if self.yolo is not None:
            if not self.yolo.update_frame(self.test_frame, use_zone=False):
                QTimer.singleShot(50, self._submit_test_frame)

    def _abort_test_batch(self, message):
        """Stop a batch that can no longer receive an inference result."""
        self.test_mode = False
        self.test_frame = None
        self.btn_save_results.setEnabled(bool(self.test_results))
        self.alert.setText(message)
        self.set_status("Test หยุดทำงาน", "#ef4444")
        self._update_test_navigation()

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
        result_image = self._decode_result_image(item["image"])
        if result_image is not None:
            self._display_frame(
                result_image,
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
                image_data = item["image"]
                if isinstance(image_data, np.ndarray):
                    image_data = self._encode_result_image(image_data)
                if image_data:
                    result_name = (
                        f"{result_index:03d}_{Path(image_name).stem}_detected.jpg"
                    )
                    (output_path / result_name).write_bytes(image_data)
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

    def close_camera(self, clear_display=True, preserve_pending=False):
        self.timer.stop()
        self.danger_alarm.stop()
        self.voice_announcer.stop()
        self.voice_announcer.set_live_source(False)
        if not preserve_pending:
            # A manual Stop or a source error cancels a queued video switch.
            self._pending_video_path = None
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
        stopped = True
        # Detach YOLO before stopping the provider so it cannot pull from a
        # capture object while that thread is being closed.
        if self.yolo is not None:
            self.yolo.clear_frame_source()
        if self.capture is not None:
            capture = self.capture
            # Do not block the Qt GUI while a camera backend is stuck in
            # read(). The retry timer below owns the eventual cleanup.
            if capture.stop(timeout=0):
                capture.deleteLater()
                self.capture = None
                self._stopping_capture = None
            else:
                # Never discard a still-running QThread. Keep the reference
                # and retry asynchronously so a blocked camera read cannot
                # freeze the UI or allow two capture workers at once.
                self._stopping_capture = capture
                QTimer.singleShot(100, self._finish_capture_stop)
                stopped = False
        self._last_capture_frame_number = -1
        self.current_frame = None
        self._reset_display_throttle()
        self.camera_badge.setText("กล้อง ปิด")
        self.camera_badge.setStyleSheet("background:#162238; color:#94A3B8; border:1px solid #26364D; border-radius:8px; padding:6px 10px; font-size:12px; font-weight:700;")
        self.source_name = "No source"
        self.source_metric.value_label.setText(self.source_name)
        self.fps_metric.value_label.setText("--")
        self.danger_count.setText("อันตราย: 0")
        self.warning_count.setText("ระวัง: 0")
        self.safe_count.setText("ปลอดภัย: 0")
        if self.yolo is not None:
            self.yolo.clear_frame()
        if clear_display:
            self.video.clear()
            self.video.setText("ยังไม่ได้เปิดกล้องหรือวิดีโอ")
            self.last_info = "<span style='color:#94a3b8;'>ปิดกล้องแล้ว</span>"
            self._last_alert_html = None
            self.alert.setHtml(self.last_info)
            if self.yolo is None:
                self.set_status("โมเดลไม่พร้อมใช้งาน", "#ef4444")
            elif not self.yolo.model_ready:
                self.set_status("กำลังเตรียมโมเดล", "#f59e0b")
            else:
                self.set_status("พร้อมใช้งาน", "#22c55e")
        return stopped

    def _on_capture_error(self, message):
        if self.sender() is not self.capture or self.capture is None:
            return
        self.close_camera(clear_display=False)
        self.alert.setText(message)
        self.set_status("แหล่งภาพผิดพลาด", "#ef4444")

    def _finish_capture_stop(self):
        capture = self._stopping_capture
        if capture is None:
            return
        if capture.isRunning():
            # Do not wait on the GUI thread. Some camera backends can remain
            # inside read() briefly even after a stop request.
            capture.stop(timeout=0)
            QTimer.singleShot(50, self._finish_capture_stop)
            return
        capture.deleteLater()
        if self.capture is capture:
            self.capture = None
        self._stopping_capture = None
        pending_video_path = self._pending_video_path
        if pending_video_path:
            # The file dialog can be used again while the old decoder is
            # stopping. Start the newest selected video now that no capture
            # worker is left behind.
            self.open_video(pending_video_path, _after_stop=True)

    def _on_capture_end(self):
        if self.sender() is not self.capture or not self.video_is_file:
            return
        self.timer.stop()
        self.video_slider.setValue(max(0, self.video_total_frames - 1))
        self.alert.setText(
            "วิดีโอจบแล้ว สามารถลากแถบเวลาเพื่อดูช่วงอื่นได้"
        )
        self.set_status("วิดีโอจบแล้ว", "#2563eb")

    def update_frame(self):
        if self.capture is None:
            return

        frame, frame_number = self.capture.get_latest()
        if frame is None or frame_number < 0:
            return
        if frame_number == self._last_capture_frame_number:
            return
        self._last_capture_frame_number = frame_number

        # CaptureThread publishes immutable frame references. Keep the raw
        # frame for zone previews and make only one drawing copy for the UI.
        self.current_frame = frame
        if self.video_is_file and not self.video_slider_dragging:
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(frame_number)
            self.video_slider.blockSignals(False)
            self._preview_video_slider_time(frame_number)
        if self.yolo is not None and not self.yolo.frame_source_attached:
            # Audio is pre-generated WAV playback and never performs TTS
            # inference at runtime, so it must not be allowed to pause YOLO.
            self.yolo.update_frame(frame)

        self._display_frame(frame, draw_zone=True, display_only=True)

    def _display_frame(
        self, frame, draw_zone, show_detections=True, display_only=False
    ):
        source_height, source_width = frame.shape[:2]
        render_scale = 1.0
        if display_only:
            # Real-time preview does not need to process every source pixel at
            # 1080p/4K. Resize once to the widget while keeping the original
            # frame untouched for inference, zone editing, and exports.
            widget_width = max(self.video.width(), 1)
            widget_height = max(self.video.height(), 1)
            if widget_width > 10 and widget_height > 10:
                render_scale = min(
                    widget_width / source_width,
                    widget_height / source_height,
                    1.0,
                )
            if render_scale < 0.999:
                render_width = max(1, int(round(source_width * render_scale)))
                render_height = max(1, int(round(source_height * render_scale)))
                frame = cv2.resize(
                    frame, (render_width, render_height), interpolation=cv2.INTER_AREA
                )
            else:
                frame = frame.copy()

        scale_x = frame.shape[1] / max(source_width, 1)
        scale_y = frame.shape[0] / max(source_height, 1)
        detections = (
            self.yolo.get_detections()
            if show_detections and self.yolo
            else []
        )
        danger_total = warning_total = safe_total = 0
        if draw_zone and hasattr(self, "danger_count"):
            danger_total = sum(d["status"] == "DANGER" for d in detections)
            warning_total = sum(d["status"] == "WARNING" for d in detections)
        if draw_zone and hasattr(self, "safe_count"):
            safe_total = sum(d["status"] == "SAFE" for d in detections)
        if draw_zone and self._last_detection_counts != (
            danger_total,
            warning_total,
            safe_total,
        ):
            self.danger_count.setText(f"อันตราย: {danger_total}")
            self.warning_count.setText(f"ระวัง: {warning_total}")
            self.safe_count.setText(f"ปลอดภัย: {safe_total}")
            self._last_detection_counts = (
                danger_total,
                warning_total,
                safe_total,
            )
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            color = detection["color"]
            # Prediction is allowed to move a box between AI results, but
            # never let a bad/fast track produce unbounded drawing coordinates.
            x1 = max(0, min(source_width - 1, int(x1)))
            y1 = max(0, min(source_height - 1, int(y1)))
            x2 = max(0, min(source_width - 1, int(x2)))
            y2 = max(0, min(source_height - 1, int(y2)))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            render_box = (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            )
            rx1, ry1, rx2, ry2 = render_box
            line_width = max(1, int(round(2 * min(scale_x, scale_y))))
            font_scale = max(0.35, 0.6 * min(scale_x, scale_y))
            text_y = max(16, ry1 - 8)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, line_width)
            cv2.putText(
                frame,
                f'{detection["label"]} {detection["dist"]} M [{detection["status"]}]',
                (rx1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                line_width,
            )

        zone = (
            self.yolo.get_zone((source_height, source_width, 3))
            if draw_zone and self.yolo
            else None
        )
        if zone is not None:
            if display_only and (scale_x != 1.0 or scale_y != 1.0):
                zone = np.column_stack(
                    (
                        np.rint(zone[:, 0] * scale_x),
                        np.rint(zone[:, 1] * scale_y),
                    )
                ).astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (30, 80, 180))
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
            cv2.polylines(frame, [zone], True, (56, 189, 248), 2)

        height, width, channels = frame.shape
        bgr_format = getattr(QImage, "Format_BGR888", None)
        if bgr_format is not None:
            # Qt 5.15 can consume OpenCV's BGR buffer directly. This removes
            # one full-frame BGR->RGB conversion on every display tick.
            image = QImage(
                frame.data,
                width,
                height,
                int(frame.strides[0]),
                bgr_format,
            ).copy()
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(
                rgb_frame.data,
                width,
                height,
                channels * width,
                QImage.Format_RGB888,
            ).copy()
        pixmap = QPixmap.fromImage(image)
        if display_only:
            # The capture buffer stays small (640x360), but the preview should
            # cover the whole card. Expand proportionally, then crop the
            # excess center area so there are no black side bars and no image
            # distortion.
            target_size = self.video.size()
            if target_size.width() > 10 and target_size.height() > 10:
                pixmap = pixmap.scaled(
                    target_size,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.FastTransformation,
                )
                crop_x = max(0, (pixmap.width() - target_size.width()) // 2)
                crop_y = max(0, (pixmap.height() - target_size.height()) // 2)
                pixmap = pixmap.copy(
                    crop_x,
                    crop_y,
                    target_size.width(),
                    target_size.height(),
                )
            self.video.setPixmap(pixmap)
        else:
            self.video.setPixmap(
                pixmap.scaled(self.video.size(), Qt.KeepAspectRatio)
            )
        if draw_zone:
            now = time.time()
            self.fps = 1.0 / max(now - self.last_time, 1e-6)
            self.last_time = now
            if now - self._last_ui_status_update >= UI_STATUS_UPDATE_INTERVAL_S:
                ai_fps = self.yolo.inference_fps if self.yolo is not None else 0.0
                self.fps_metric.value_label.setText(
                    f"{self.fps:.1f} / {ai_fps:.1f}"
                )
                if hasattr(self, "bottom_status"):
                    model_name = self.model_name
                    device_name = (
                        "GPU"
                        if self.yolo is not None and self.yolo.device == "cuda"
                        else "CPU"
                    )
                    audio_state = "เปิด" if self.voice_announcer.enabled else "ปิด"
                    self.bottom_status.setText(
                        f"แหล่งภาพ: {self.source_name} | โมเดล: {model_name} | "
                        f"FPS: {self.fps:.1f} | AI: {device_name} | โซน: เปิด | เสียง: {audio_state}"
                    )
                self._last_ui_status_update = now
            if self.last_info != self._last_alert_html:
                self.alert.setHtml(self.last_info)
                self._last_alert_html = self.last_info
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
            # DANGER is prioritized when both levels exist.  When only
            # WARNING objects remain, use the quieter/slower warning siren.
            self.danger_alarm.set_active(
                nearest_alert is not None,
                nearest_alert["status"] if nearest_alert else "SAFE",
            )
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
                    "image": self._encode_result_image(annotated),
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
        self.alert.setHtml(self.last_info)
        failed_worker = self.sender()
        if (
            message.startswith("YOLO model load error:")
            and failed_worker is self.yolo
        ):
            # A QThread that returned from run() cannot accept frames again.
            # Clear it immediately so the UI never waits forever for a result.
            self.yolo = None
            if failed_worker is not None:
                failed_worker.deleteLater()
            if self.test_mode:
                self._abort_test_batch("โมเดลโหลดไม่สำเร็จ จึงหยุดการ Test")
        if self.test_mode:
            self.alert.setHtml(self.last_info)
            if self.test_index < len(self.test_queue):
                self.test_results.append(
                    {
                        "path": self.test_queue[self.test_index],
                        "detections": [],
                        "image": self._encode_result_image(self.test_frame)
                        if self.test_frame is not None
                        else None,
                        "error": True,
                        "info": self.last_info,
                    }
                )
                self.test_index += 1
                QTimer.singleShot(50, self._load_next_test_image)
        self.set_status("ตรวจจับผิดพลาด", "#ef4444")

    def on_yolo_ready(self):
        """Show ready only after model load, fuse, and warm-up are complete."""
        if self.sender() is self.yolo:
            # Keep CPU inference below the UI, but do not leave a CUDA worker
            # at LowestPriority after it has finished loading.  GPU inference
            # is mostly asynchronous and benefits from normal scheduling.
            priority = (
                QThread.NormalPriority
                if self.yolo.device == "cuda"
                else QThread.LowestPriority
            )
            self.yolo.setPriority(priority)
            self.set_status("พร้อมใช้งาน", "#22c55e")

    def open_settings(self):
        image_path = self._zone_preview_path()
        if self.settings_win is None:
            self.settings_win = SettingsWindow(str(image_path) if image_path else None)
            self.settings_win.settings_saved.connect(self.on_settings_saved)
            self.settings_win.zone_saved.connect(self._invalidate_zone)
        else:
            self.settings_win.set_zone_image(str(image_path) if image_path else None)
        self.settings_win.setWindowModality(Qt.ApplicationModal)
        self.settings_win.show()
        self.settings_win.raise_()
        self.settings_win.activateWindow()

    def on_settings_saved(self, kind, value):
        if kind == "distance":
            self.on_thresholds_saved(value["danger"], value["warning"])
        elif kind == "camera":
            self.on_camera_settings_saved(
                value["camera_index"],
                value["focal_length"],
                value.get("object_heights"),
            )
        elif kind == "audio":
            self.on_alert_settings_saved(
                value["siren_enabled"],
                value["voice_enabled"],
                value["voice_model"],
                value.get("siren_volume", 100),
                value.get("voice_volume", 100),
            )
        elif kind == "model":
            self.switch_model(value)

    def on_camera_settings_saved(self, camera_index, focal_length, object_heights=None):
        if self.yolo is not None:
            self.yolo.update_camera_settings(focal_length)
            if object_heights:
                self.yolo.update_object_heights(object_heights)
        self.last_info = (
            f"<span style='color:#e5e7eb;'>บันทึกค่าคำนวณระยะแล้ว: "
            f"Focal Length {focal_length:.1f} px และความสูงวัตถุ</span>"
        )
        self.alert.setHtml(self.last_info)
        self._last_alert_html = self.last_info

    def switch_model(self, relative_path):
        """Restart the detection thread on the newly selected model."""
        if not self.close_camera(clear_display=False):
            self.set_status("กำลังปิดแหล่งภาพก่อนเปลี่ยนโมเดล", "#f59e0b")
            QTimer.singleShot(100, lambda: self.switch_model(relative_path))
            return
        self.model_name = relative_path

        model_relative, model_path = load_model_settings()
        self._pending_model = (model_relative, model_path)
        if self._stopping_yolo is not None:
            self.set_status("กำลังเปลี่ยนโมเดล", "#f59e0b")
            return
        if self.yolo is not None:
            old_yolo = self.yolo
            self.yolo = None
            old_yolo.result_ready.disconnect(self.on_yolo_result)
            old_yolo.error_ready.disconnect(self.on_yolo_error)
            old_yolo.model_ready_signal.disconnect(self.on_yolo_ready)
            self._stopping_yolo = old_yolo
            old_yolo.finished.connect(self._finish_model_switch)
            self.set_status("กำลังเปลี่ยนโมเดล", "#f59e0b")
            if old_yolo.isRunning():
                old_yolo.stop(wait=False)
            else:
                self._finish_model_switch()
            return

        self._start_yolo_model(model_relative, model_path)

    def _finish_model_switch(self):
        if self.sender() is not None and self.sender() is not self._stopping_yolo:
            return
        old_yolo = self._stopping_yolo
        self._stopping_yolo = None
        if old_yolo is not None:
            old_yolo.deleteLater()
        pending_model = self._pending_model
        self._pending_model = None
        if pending_model is not None:
            self._start_yolo_model(*pending_model)

    def _start_yolo_model(self, model_relative, model_path):
        self.model_name = model_relative
        try:
            self.yolo = YoloThread(model_path=model_path)
            self.yolo.result_ready.connect(self.on_yolo_result)
            self.yolo.error_ready.connect(self.on_yolo_error)
            self.yolo.model_ready_signal.connect(self.on_yolo_ready)
            self.yolo.start(QThread.LowestPriority)
            self.last_info = (
                f"<span style='color:#f59e0b;'>กำลังเตรียมโมเดล \"{model_relative}\"...</span>"
            )
            self.alert.setHtml(self.last_info)
            self._last_alert_html = self.last_info
            self.set_status("กำลังเตรียมโมเดล", "#f59e0b")
        except Exception as error:
            self.alert.setText(f"Cannot load YOLO model ({model_relative}): {error}")
            self.set_status("โหลดโมเดลไม่สำเร็จ", "#ef4444")

    def on_alert_settings_saved(
        self,
        siren_enabled,
        voice_enabled,
        voice_model,
        siren_volume=100,
        voice_volume=100,
    ):
        self.danger_alarm.set_enabled(siren_enabled)
        self.danger_alarm.set_volume(siren_volume)
        self.voice_announcer.set_voice_model(voice_model)
        self.voice_announcer.set_enabled(voice_enabled)
        self.voice_announcer.set_volume(voice_volume)
        self._refresh_mute_button()
        self.last_info = (
            "<span style='color:#e5e7eb;'>บันทึกการตั้งค่าเสียงแจ้งเตือนแล้ว: "
            f"ไซเรน {'เปิด' if siren_enabled else 'ปิด'}, "
            f"เสียงพูด {'เปิด' if voice_enabled else 'ปิด'}</span>"
        )
        self._last_alert_html = None
        self.alert.setHtml(self.last_info)

    def on_thresholds_saved(self, danger, warning):
        self.update_distance_status()
        if self.yolo is not None:
            self.yolo.update_thresholds(danger, warning)
        self.last_info = (
            f"<span style='color:#e5e7eb;'>บันทึกระยะแล้ว: "
            f"อันตราย ≤ {danger} M, ระวัง ≤ {warning} M</span>"
        )

    def _invalidate_zone(self):
        """Invalidate the currently active detector after a zone save."""
        if self.yolo is not None and self.yolo.isRunning():
            self.yolo.invalidate_zone()

    def _zone_preview_path(self):
        """Save the latest frame and return the preview path when available."""
        TEMP_DIR.mkdir(exist_ok=True)
        image_path = TEMP_DIR / "zone_preview.jpg"
        if self.current_frame is None:
            return None
        if not cv2.imwrite(str(image_path), self.current_frame):
            return None
        return image_path

    def closeEvent(self, event):
        self.close_camera()
        if self.capture is not None and self.capture.isRunning():
            event.ignore()
            QTimer.singleShot(250, self.close)
            return
        self.danger_alarm.stop()
        self.voice_announcer.stop()
        if self.yolo is not None and self.yolo.isRunning():
            # Inference can be inside a model call; stopping it must not
            # block the Qt event loop while the window is closing.
            self.yolo.stop(wait=False)
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        if self._stopping_yolo is not None and self._stopping_yolo.isRunning():
            self._stopping_yolo.stop(wait=False)
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        event.accept()
