from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vision.alert_config import load_alert_settings, save_alert_settings


APP_STYLE = """
QMainWindow { background: #0f172a; }
QLabel { color: #e5e7eb; }
QFrame#card {
    background: #172033;
    border: 1px solid #2d3a4f;
    border-radius: 18px;
}
QCheckBox {
    color: #e5e7eb;
    font-size: 15px;
    font-weight: 700;
    spacing: 10px;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 5px;
    border: 1px solid #475569;
    background: #0b1220;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border: 1px solid #2563eb;
}
QPushButton {
    background: #2563eb; color: white; border: none;
    border-radius: 12px; padding: 11px; font-size: 15px; font-weight: 700;
}
QPushButton:hover { background: #1d4ed8; }
"""


class AlertSettingWindow(QMainWindow):
    settings_saved = pyqtSignal(bool, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye - ตั้งค่าเสียงแจ้งเตือน")
        self.resize(520, 380)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)

        title = QLabel("รูปแบบเสียงแจ้งเตือนเมื่ออยู่ในระยะอันตราย")
        title.setFont(QFont("Arial", 17, QFont.Bold))
        hint = QLabel(
            "เลือกได้ว่าจะใช้เสียงไซเรน เสียงพูดบอกชนิดวัตถุ หรือทั้งสองอย่าง "
            "(ถ้าไม่เลือกเลย จะไม่มีเสียงแจ้งเตือนแม้จะไม่ได้ปิดเสียงหลักไว้)"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8; font-size:13px;")

        card = QFrame(objectName="card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(16)

        self.siren_check = QCheckBox("🚨  เสียงไซเรน (สัญญาณเตือน)")
        self.voice_check = QCheckBox("🗣️  เสียงพูดแจ้งเตือน (ระบุชนิดวัตถุ)")
        card_layout.addWidget(self.siren_check)
        card_layout.addWidget(self.voice_check)

        freq_note = QLabel(
            "หมายเหตุ: เสียงพูดจะไม่พูดซ้ำวัตถุเดิม เว้นแต่ระยะห่างเปลี่ยนไปจากจุดที่แจ้งครั้งก่อน "
            "อย่างน้อย 3 เมตร เพื่อลดความถี่การแจ้งเตือน"
        )
        freq_note.setWordWrap(True)
        freq_note.setStyleSheet(
            "background:#0b1220; color:#cbd5e1; border-radius:10px; "
            "padding:12px; font-size:12px; font-weight:400;"
        )
        card_layout.addWidget(freq_note)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#facc15; font-size:13px;")
        card_layout.addWidget(self.status)

        save_button = QPushButton("บันทึกการตั้งค่า")
        save_button.setFixedHeight(46)
        save_button.clicked.connect(self.save_values)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(card)
        layout.addStretch()
        layout.addWidget(save_button)

    def _load_values(self):
        settings = load_alert_settings()
        self.siren_check.setChecked(settings["siren_enabled"])
        self.voice_check.setChecked(settings["voice_enabled"])

    def save_values(self):
        settings = save_alert_settings(
            self.siren_check.isChecked(), self.voice_check.isChecked()
        )
        self.settings_saved.emit(settings["siren_enabled"], settings["voice_enabled"])
        self.close()
