"""Single-window settings experience for Third Eye.

The page widgets use the existing configuration modules and emit small,
typed results back to MainWindow. Detection and audio engines remain outside
this module.
"""

from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.icons import icon
from vision.alert_config import load_alert_settings, save_alert_settings
from vision.camera_config import load_camera_settings, save_camera_settings
from vision.distance_config import load_distance_thresholds, save_distance_thresholds
from vision.model_config import (
    import_model_file,
    list_available_models,
    load_model_settings,
    save_model_settings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZONE_DIR = PROJECT_ROOT / "zones"
ACTIVE_ZONE = ZONE_DIR / "active.txt"
DEFAULT_ZONE = ZONE_DIR / "default.txt"
VIEW_W = 800
VIEW_H = 480


STYLE = """
QMainWindow, QWidget { background:#0B1220; color:#F8FAFC; font-family:'Leelawadee UI','Segoe UI'; }
QFrame#settingsHeader { background:#111C2E; border-bottom:1px solid #26364D; }
QFrame#sidebar { background:#111C2E; border-right:1px solid #26364D; }
QFrame#page { background:#0B1220; }
QFrame#card { background:#162238; border:1px solid #26364D; border-radius:12px; }
QLabel#title { color:#F8FAFC; font-size:23px; font-weight:700; }
QLabel#section { color:#F8FAFC; font-size:17px; font-weight:700; }
QLabel#hint, QLabel#muted { color:#94A3B8; font-size:12px; }
QLabel#status { color:#22C55E; font-size:12px; font-weight:700; }
QListWidget { background:#111C2E; border:0; outline:0; padding:12px 8px; }
QListWidget::item { color:#CBD5E1; padding:12px 14px; margin:2px 0; border-radius:8px; }
QListWidget::item:selected { background:#1D4ED8; color:#FFFFFF; }
QListWidget::item:hover { background:#1E3A5F; }
QComboBox, QSpinBox, QDoubleSpinBox { background:#0B1220; color:#F8FAFC; border:1px solid #334155; border-radius:8px; padding:9px 10px; min-height:22px; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border:1px solid #3B82F6; }
QComboBox QAbstractItemView { background:#111C2E; color:#F8FAFC; selection-background-color:#2563EB; }
QCheckBox { color:#F8FAFC; font-size:14px; spacing:10px; padding:8px 0; }
QCheckBox::indicator { width:18px; height:18px; border:1px solid #64748B; border-radius:5px; background:#0B1220; }
QCheckBox::indicator:checked { background:#2563EB; border:1px solid #60A5FA; }
QPushButton { background:#334155; color:#F8FAFC; border:0; border-radius:8px; padding:10px 16px; font-size:13px; font-weight:700; }
QPushButton:hover { background:#475569; }
QPushButton#primary { background:#2563EB; color:#FFFFFF; }
QPushButton#primary:hover { background:#3B82F6; }
QPushButton#quiet { background:transparent; color:#CBD5E1; border:1px solid #334155; }
QPushButton#quiet:hover { background:#162238; }
QScrollArea { border:0; background:#0B1220; }
"""


class BasePage(QWidget):
    def __init__(self, title, description):
        super().__init__()
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(32, 26, 32, 22)
        self.body.setSpacing(16)
        heading = QLabel(title)
        heading.setObjectName("title")
        description_label = QLabel(description)
        description_label.setObjectName("hint")
        description_label.setWordWrap(True)
        self.body.addWidget(heading)
        self.body.addWidget(description_label)

    def apply(self):
        return None


class DistancePage(BasePage):
    def __init__(self):
        super().__init__("ระยะเตือนภัย", "กำหนดขอบเขต อันตราย ระวัง และปลอดภัย ระบบจะใช้ค่าจริงนี้ในการแจ้งเตือน")
        card = QFrame(objectName="card")
        form = QFormLayout(card)
        form.setContentsMargins(22, 20, 22, 20)
        form.setSpacing(16)
        self.danger = self._spin()
        self.warning = self._spin()
        form.addRow("ระยะอันตราย", self.danger)
        form.addRow("ระยะระวัง", self.warning)
        self.body.addWidget(card)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("background:#111C2E; border:1px solid #26364D; border-radius:8px; padding:14px; color:#CBD5E1;")
        self.body.addWidget(self.preview)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.body.addWidget(self.status)
        self.danger.valueChanged.connect(self.update_preview)
        self.warning.valueChanged.connect(self.update_preview)
        values = load_distance_thresholds()
        self.danger.setValue(values["danger"])
        self.warning.setValue(values["warning"])
        self.update_preview()
        self.body.addStretch()

    @staticmethod
    def _spin():
        box = QDoubleSpinBox()
        box.setRange(0.1, 200.0)
        box.setDecimals(1)
        box.setSingleStep(0.5)
        box.setSuffix(" เมตร")
        return box

    def update_preview(self):
        danger, warning = self.danger.value(), self.warning.value()
        if warning <= danger:
            self.preview.setText("ค่าระยะไม่ถูกต้อง: ระยะระวังต้องมากกว่าระยะอันตราย")
            self.preview.setStyleSheet("background:#3B1D2A; border:1px solid #EF4444; border-radius:8px; padding:14px; color:#FECACA;")
            return
        self.preview.setStyleSheet("background:#111C2E; border:1px solid #26364D; border-radius:8px; padding:14px; color:#CBD5E1;")
        self.preview.setText(
            f"ตัวอย่างการแบ่งระดับ\nอันตราย  ≤ {danger:.1f} เมตร    |    "
            f"ระวัง  {danger:.1f}–{warning:.1f} เมตร    |    ปลอดภัย  > {warning:.1f} เมตร"
        )

    def apply(self):
        if self.warning.value() <= self.danger.value():
            self.status.setStyleSheet("color:#F87171; font-size:12px; font-weight:700;")
            self.status.setText("บันทึกไม่ได้: ระยะระวังต้องมากกว่าระยะอันตราย")
            return None
        result = save_distance_thresholds(self.danger.value(), self.warning.value())
        self.status.setStyleSheet("color:#22C55E; font-size:12px; font-weight:700;")
        self.status.setText("บันทึกระยะเตือนภัยแล้ว")
        return result


class CameraPage(BasePage):
    def __init__(self):
        super().__init__("กล้องและการคำนวณระยะ", "เลือกกล้องและปรับ Focal Length ให้ตรงกับอุปกรณ์จริงเพื่อให้ค่าระยะมีความแม่นยำขึ้น")
        card = QFrame(objectName="card")
        form = QFormLayout(card)
        form.setContentsMargins(22, 20, 22, 20)
        form.setSpacing(16)
        self.camera = QSpinBox()
        self.camera.setRange(0, 20)
        self.camera.setToolTip("0 คือกล้องตัวแรก, 1 คือกล้องตัวถัดไป")
        self.focal = QDoubleSpinBox()
        self.focal.setRange(1, 10000)
        self.focal.setDecimals(1)
        self.focal.setSingleStep(10)
        self.focal.setSuffix(" px")
        form.addRow("หมายเลขกล้อง", self.camera)
        form.addRow("Focal Length", self.focal)
        self.body.addWidget(card)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.body.addWidget(self.status)
        values = load_camera_settings()
        self.camera.setValue(values["camera_index"])
        self.focal.setValue(values["focal_length"])
        self.body.addStretch()

    def apply(self):
        result = save_camera_settings(self.camera.value(), self.focal.value())
        self.status.setText("บันทึกค่ากล้องแล้ว")
        return result


class ModelPage(BasePage):
    def __init__(self):
        super().__init__("โมเดลตรวจจับ", "เลือกโมเดล YOLO ที่จะใช้ประมวลผล หรือเพิ่มไฟล์ .pt จากเครื่องนี้")
        card = QFrame(objectName="card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        self.combo = QComboBox()
        self.add_button = QPushButton("เลือกไฟล์โมเดลจากเครื่อง")
        self.add_button.setIcon(icon("image"))
        self.add_button.clicked.connect(self.add_model)
        layout.addWidget(QLabel("โมเดลที่ใช้งาน"))
        layout.addWidget(self.combo)
        layout.addWidget(self.add_button)
        self.body.addWidget(card)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.body.addWidget(self.status)
        self.initial, _ = load_model_settings()
        self.reload()
        self.body.addStretch()

    def reload(self, selected=None):
        current, _ = load_model_settings()
        selected = selected or current
        self.combo.clear()
        models = list_available_models()
        for item in models:
            self.combo.addItem(item["label"], item["path"])
        index = self.combo.findData(selected)
        if index >= 0:
            self.combo.setCurrentIndex(index)

    def add_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์โมเดล", "", "YOLO Model (*.pt)")
        if not path:
            return
        try:
            relative = import_model_file(path)
        except (ValueError, OSError) as error:
            self.status.setStyleSheet("color:#F87171; font-size:12px; font-weight:700;")
            self.status.setText(str(error))
            return
        self.reload(relative)
        self.status.setStyleSheet("color:#22C55E; font-size:12px; font-weight:700;")
        self.status.setText(f"เพิ่มโมเดล {relative} แล้ว")

    def apply(self):
        relative = self.combo.currentData()
        if not relative:
            self.status.setText("ไม่พบโมเดล .pt")
            return None
        save_model_settings(relative)
        return {"path": relative, "changed": relative != self.initial}


class AudioPage(BasePage):
    def __init__(self):
        super().__init__("เสียงแจ้งเตือน", "เปิดใช้ไซเรนหรือเสียงพูด และเลือกโมเดลเสียงภาษาไทยที่ติดตั้งอยู่ในโปรเจกต์")
        card = QFrame(objectName="card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(8)
        self.siren = QCheckBox("เสียงไซเรนเมื่ออยู่ในระยะอันตราย")
        self.voice = QCheckBox("เสียงพูดแจ้งเตือนพร้อมชนิดวัตถุและระยะ")
        self.combo = QComboBox()
        voice_dir = PROJECT_ROOT / "tts" / "voices"
        names = {"th_f_1":"เสียงผู้หญิง 1", "th_m_1":"เสียงผู้ชาย 1", "th_f_2":"เสียงผู้หญิง 2"}
        for path in sorted(voice_dir.glob("th_*.onnx")):
            self.combo.addItem(f"{names.get(path.stem, path.stem)} ({path.stem})", path.stem)
        layout.addWidget(self.siren)
        layout.addWidget(self.voice)
        layout.addWidget(QLabel("โมเดลเสียงภาษาไทย"))
        layout.addWidget(self.combo)
        self.body.addWidget(card)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.body.addWidget(self.status)
        values = load_alert_settings()
        self.siren.setChecked(values["siren_enabled"])
        self.voice.setChecked(values["voice_enabled"])
        index = self.combo.findData(values["voice_model"])
        if index >= 0:
            self.combo.setCurrentIndex(index)
        self.body.addStretch()

    def apply(self):
        result = save_alert_settings(self.siren.isChecked(), self.voice.isChecked(), self.combo.currentData() or "th_m_1")
        self.status.setText("บันทึกการตั้งค่าเสียงแล้ว")
        return result


class ZonePage(BasePage):
    zone_saved = pyqtSignal()

    def __init__(self, image_path=None):
        super().__init__("โซนตรวจจับ", "ลากจุดควบคุมบนภาพเพื่อกำหนดพื้นที่ที่ต้องการตรวจจับ")
        self.image = self._load_image(image_path)
        self.points = [[80, 430], [400, 140], [720, 430]]
        self.drag_index = None
        content = QHBoxLayout()
        content.setSpacing(16)
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setMinimumSize(620, 360)
        self.image_label.setStyleSheet("background:#020617; border:1px solid #26364D; border-radius:10px;")
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release
        side = QFrame(objectName="card")
        right = QVBoxLayout(side)
        right.setContentsMargins(18, 18, 18, 18)
        right.setSpacing(12)
        self.preset = QComboBox()
        self.load_presets()
        self.preset.currentTextChanged.connect(self.load_zone)
        right.addWidget(QLabel("รูปแบบโซน"))
        right.addWidget(self.preset)
        instruction = QLabel("ลากจุดสีขาวทั้ง 3 จุดเพื่อกำหนดพื้นที่ตรวจจับ\nจากนั้นกด บันทึกการตั้งค่า ด้านล่าง")
        instruction.setObjectName("muted")
        instruction.setWordWrap(True)
        right.addWidget(instruction)
        right.addStretch()
        self.zone_status = QLabel("")
        self.zone_status.setObjectName("status")
        right.addWidget(self.zone_status)
        content.addWidget(self.image_label, 3)
        content.addWidget(side, 1)
        self.body.addLayout(content, 1)
        self.load_active_zone()
        self.update_view()

    @staticmethod
    def _load_image(image_path):
        image = cv2.imread(str(image_path)) if image_path and Path(image_path).is_file() else None
        if image is None:
            image = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        return cv2.resize(image, (VIEW_W, VIEW_H))

    def update_view(self):
        frame = self.image.copy()
        overlay = frame.copy()
        polygon = np.array(self.points, np.int32)
        cv2.fillPoly(overlay, [polygon], (30, 80, 180))
        frame = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)
        cv2.polylines(frame, [polygon], True, (56, 189, 248), 3)
        for index, (x, y) in enumerate(self.points, 1):
            cv2.circle(frame, (x, y), 12, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 12, (56, 189, 248), 2)
            cv2.putText(frame, str(index), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 23, 42), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, VIEW_W, VIEW_H, 3 * VIEW_W, QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _image_position(self, event):
        """Convert the displayed, aspect-fit image coordinate to 800x480."""
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        scale = min(
            self.image_label.width() / VIEW_W,
            self.image_label.height() / VIEW_H,
        )
        drawn_width = VIEW_W * scale
        drawn_height = VIEW_H * scale
        offset_x = (self.image_label.width() - drawn_width) / 2
        offset_y = (self.image_label.height() - drawn_height) / 2
        x = (event.pos().x() - offset_x) / scale
        y = (event.pos().y() - offset_y) / scale
        if not (0 <= x < VIEW_W and 0 <= y < VIEW_H):
            return None
        return int(x), int(y)

    def mouse_press(self, event):
        position = self._image_position(event)
        if position is None:
            return
        x, y = position
        for index, (point_x, point_y) in enumerate(self.points):
            if abs(point_x - x) < 24 and abs(point_y - y) < 24:
                self.drag_index = index
                return

    def mouse_move(self, event):
        if self.drag_index is None:
            return
        position = self._image_position(event)
        if position is None:
            return
        x, y = position
        self.points[self.drag_index] = [
            max(0, min(x, VIEW_W - 1)),
            max(0, min(y, VIEW_H - 1)),
        ]
        self.update_view()

    def mouse_release(self, event):
        self.drag_index = None

    @staticmethod
    def read_zone(path):
        points = []
        try:
            with Path(path).open(encoding="utf-8") as file:
                for line in file:
                    x, y = map(float, line.strip().split(","))
                    if not 0 <= x <= 1 or not 0 <= y <= 1:
                        return None
                    points.append([int(x * VIEW_W), int(y * VIEW_H)])
        except (OSError, ValueError):
            return None
        if len(points) < 3 or len(set(map(tuple, points))) != len(points):
            return None
        if abs(cv2.contourArea(np.array(points, np.int32))) < 100:
            return None
        return points

    def load_presets(self):
        ZONE_DIR.mkdir(exist_ok=True)
        presets = sorted(path.name for path in ZONE_DIR.glob("*.txt") if path.name != ACTIVE_ZONE.name and path.is_file())
        if not presets:
            presets = [DEFAULT_ZONE.name]
        self.preset.addItems(presets)

    def load_active_zone(self):
        points = self.read_zone(ACTIVE_ZONE)
        if points:
            self.points = points

    def load_zone(self, name):
        points = self.read_zone(ZONE_DIR / name)
        if points:
            self.points = points
            self.update_view()

    def apply(self):
        polygon = np.array(self.points, np.int32)
        if len(set(map(tuple, self.points))) < 3 or abs(cv2.contourArea(polygon)) < 100:
            self.zone_status.setStyleSheet("color:#F87171; font-size:12px; font-weight:700;")
            self.zone_status.setText("โซนต้องมีพื้นที่และจุดอย่างน้อย 3 จุด")
            return None
        name = self.preset.currentText() or DEFAULT_ZONE.name
        target = ZONE_DIR / name
        if target != DEFAULT_ZONE:
            self._save_file(target)
        self._save_file(ACTIVE_ZONE)
        self.zone_status.setText("บันทึกโซนแล้ว")
        self.zone_saved.emit()
        return True

    def _save_file(self, path):
        ZONE_DIR.mkdir(exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as file:
            for x, y in self.points:
                file.write(f"{x / VIEW_W:.6f},{y / VIEW_H:.6f}\n")


class SettingsWindow(QMainWindow):
    settings_saved = pyqtSignal(str, object)
    zone_saved = pyqtSignal()

    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("Third Eye - ตั้งค่าระบบ")
        self.resize(1180, 760)
        self.setMinimumSize(980, 650)
        self.setStyleSheet(STYLE)
        self.pages = []
        self._build_ui(image_path)

    def _build_ui(self, image_path):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame(objectName="settingsHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 16)
        title = QLabel("ตั้งค่าระบบ")
        title.setObjectName("title")
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("ตั้งค่าทั้งหมดในหน้าต่างเดียว"))
        layout.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 16, 10, 16)
        self.navigation = QListWidget()
        side_layout.addWidget(self.navigation)
        body.addWidget(sidebar)
        self.stack = QStackedWidget()
        self.stack.setObjectName("page")
        page_specs = [
            ("ระยะเตือนภัย", "ruler", DistancePage()),
            ("โซนตรวจจับ", "scan", ZonePage(image_path)),
            ("กล้อง", "camera", CameraPage()),
            ("โมเดลตรวจจับ", "cpu", ModelPage()),
            ("เสียงแจ้งเตือน", "volume", AudioPage()),
        ]
        self.pages = [page for _, _, page in page_specs]
        for title_text, icon_name, page in page_specs:
            item = QListWidgetItem(icon(icon_name), title_text)
            self.navigation.addItem(item)
            self.stack.addWidget(page)
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        self.zone_page = self.pages[1]
        self.zone_page.zone_saved.connect(self.zone_saved.emit)
        body.addWidget(self.stack, 1)
        wrapper = QWidget()
        wrapper.setLayout(body)
        layout.addWidget(wrapper, 1)

        footer = QFrame(objectName="settingsHeader")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 12, 20, 12)
        self.status = QLabel("")
        self.status.setObjectName("status")
        footer_layout.addWidget(self.status)
        footer_layout.addStretch()
        cancel = QPushButton("ยกเลิก")
        cancel.setObjectName("quiet")
        cancel.clicked.connect(self.close)
        save = QPushButton("บันทึกการตั้งค่า")
        save.setObjectName("primary")
        save.setIcon(icon("save", "#FFFFFF"))
        save.clicked.connect(self.apply_all)
        footer_layout.addWidget(cancel)
        footer_layout.addWidget(save)
        layout.addWidget(footer)
        self.setCentralWidget(root)

    def apply_all(self):
        for page in self.pages:
            result = page.apply()
            if result is None:
                self.status.setText("กรุณาตรวจสอบค่าที่กรอกก่อนบันทึก")
                return
            if isinstance(page, ModelPage) and result.get("changed"):
                self.settings_saved.emit("model", result["path"])
                page.initial = result["path"]
            elif isinstance(page, DistancePage):
                self.settings_saved.emit("distance", result)
            elif isinstance(page, CameraPage):
                self.settings_saved.emit("camera", result)
            elif isinstance(page, AudioPage):
                self.settings_saved.emit("audio", result)
        self.status.setText("บันทึกการตั้งค่าแล้ว")
