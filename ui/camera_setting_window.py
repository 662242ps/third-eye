from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vision.camera_config import load_camera_settings, save_camera_settings


APP_STYLE = """
QMainWindow { background: #0f172a; }
QWidget { font-family: 'Leelawadee UI'; }
QLabel { color: #e5e7eb; }
QFrame#card {
    background: #172033;
    border: 1px solid #2d3a4f;
    border-radius: 18px;
}
QSpinBox, QDoubleSpinBox {
    background: #0b1220; color: #f8fafc;
    border: 1px solid #334155; border-radius: 10px;
    padding: 8px 10px; font-size: 15px; min-height: 24px;
}
QPushButton {
    background: #2563eb; color: white; border: none;
    border-radius: 12px; padding: 11px; font-size: 15px; font-weight: 700;
}
QPushButton:hover { background: #1d4ed8; }
"""


class CameraSettingWindow(QMainWindow):
    settings_saved = pyqtSignal(int, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye - ตั้งค่ากล้อง")
        self.resize(520, 390)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)

        title = QLabel("ตั้งค่ากล้องและการคำนวณระยะทาง")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        hint = QLabel(
            "ค่า Focal Length ใช้ในสูตร ระยะทาง = "
            "(ความสูงจริง × Focal Length) ÷ ความสูงวัตถุในภาพ\n"
            "ควรคาลิเบรตกับกล้องและความละเอียดที่ใช้งานจริง"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8; font-size:13px;")

        card = QFrame(objectName="card")
        form = QFormLayout(card)
        form.setContentsMargins(22, 22, 22, 22)
        form.setSpacing(18)

        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 20)
        self.camera_index.setToolTip("0 คือกล้องตัวแรก, 1 คือกล้องตัวถัดไป")
        self.focal_length = QDoubleSpinBox()
        self.focal_length.setRange(1.0, 10000.0)
        self.focal_length.setDecimals(1)
        self.focal_length.setSingleStep(10.0)
        self.focal_length.setSuffix(" px")

        form.addRow("หมายเลขกล้อง", self.camera_index)
        form.addRow("Focal Length", self.focal_length)

        save_button = QPushButton("บันทึก")
        save_button.clicked.connect(self.save_values)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(card)
        layout.addStretch()
        layout.addWidget(save_button)

    def _load_values(self):
        settings = load_camera_settings()
        self.camera_index.setValue(settings["camera_index"])
        self.focal_length.setValue(settings["focal_length"])

    def save_values(self):
        settings = save_camera_settings(
            self.camera_index.value(), self.focal_length.value()
        )
        self.settings_saved.emit(
            settings["camera_index"], settings["focal_length"]
        )
        self.close()
