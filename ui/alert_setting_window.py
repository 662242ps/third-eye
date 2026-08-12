from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vision.alert_config import load_alert_settings, save_alert_settings
from pathlib import Path


APP_STYLE = """
QMainWindow { background: #0f172a; }
QWidget { font-family: 'Leelawadee UI'; }
QComboBox { min-height: 28px; padding: 5px 10px; }
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
    settings_saved = pyqtSignal(bool, bool, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye - ตั้งค่าเสียงแจ้งเตือน")
        self.resize(520, 380)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self.setWindowTitle("Third Eye - ตั้งค่าเสียงแจ้งเตือน")
        self._load_values()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)

        title = QLabel("รูปแบบเสียงแจ้งเตือนเมื่อพบวัตถุใกล้")
        title.setFont(QFont("Arial", 17, QFont.Bold))
        hint = QLabel(
            "เลือกเสียงไซเรนสำหรับระดับอันตราย และเสียงพูดสำหรับระดับอันตราย/ระวัง "
            "โดยเสียงพูดจะแจ้งชนิดวัตถุและระยะทาง (ระดับปลอดภัยจะไม่มีเสียง)"
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

        card_layout.addWidget(QLabel("โมเดลเสียงภาษาไทย"))
        self.voice_model_combo = QComboBox()
        voice_dir = Path(__file__).resolve().parents[1] / "tts" / "voices"
        for model_path in sorted(voice_dir.glob("th_*.onnx")):
            self.voice_model_combo.addItem(model_path.stem, model_path.stem)
        card_layout.addWidget(self.voice_model_combo)

        freq_note = QLabel(
            "หมายเหตุ: ระบบจะรอผลตรวจจับให้คงอยู่ช่วงสั้น ๆ และไม่พูดซ้ำวัตถุเดิมบ่อยเกินไป "
            "เว้นแต่ระยะห่างเปลี่ยนไปอย่างน้อย 3 เมตร"
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
        index = self.voice_model_combo.findData(settings["voice_model"])
        if index >= 0:
            self.voice_model_combo.setCurrentIndex(index)

    def save_values(self):
        settings = save_alert_settings(
            self.siren_check.isChecked(),
            self.voice_check.isChecked(),
            self.voice_model_combo.currentData() or "th_f_2",
        )
        self.settings_saved.emit(
            settings["siren_enabled"],
            settings["voice_enabled"],
            settings["voice_model"],
        )
        self.close()

    def _load_values(self):
        settings = load_alert_settings()
        self.siren_check.setChecked(settings["siren_enabled"])
        self.voice_check.setChecked(settings["voice_enabled"])
        display_names = {
            "th_f_1": "เสียงผู้หญิง 1 (th_f_1)",
            "th_f_2": "เสียงผู้หญิง 2 (th_f_2)",
            "th_m_1": "เสียงผู้ชาย 1 (th_m_1)",
        }
        for index in range(self.voice_model_combo.count()):
            model_id = self.voice_model_combo.itemData(index)
            self.voice_model_combo.setItemText(index, display_names.get(model_id, model_id))
        index = self.voice_model_combo.findData(settings["voice_model"])
        if index >= 0:
            self.voice_model_combo.setCurrentIndex(index)
