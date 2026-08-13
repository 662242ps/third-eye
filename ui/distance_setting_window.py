from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vision.distance_config import load_distance_thresholds, save_distance_thresholds


APP_STYLE = """
QMainWindow {
    background: #0f172a;
}
QWidget { font-family: 'Leelawadee UI'; }
QLabel {
    color: #e5e7eb;
}
QFrame#header {
    background: #111827;
    border-bottom: 1px solid #263143;
}
QFrame#card {
    background: #172033;
    border: 1px solid #2d3a4f;
    border-radius: 18px;
}
QDoubleSpinBox {
    background: #0b1220;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 8px 10px;
    font-size: 16px;
    min-height: 24px;
}
QDoubleSpinBox:focus {
    border: 1px solid #38bdf8;
}
QPushButton#primaryButton {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 16px;
    font-weight: 700;
}
QPushButton#primaryButton:hover {
    background: #1d4ed8;
}
QPushButton#primaryButton:pressed {
    background: #1e40af;
}
"""


class DistanceSettingWindow(QMainWindow):
    thresholds_saved = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye - ตั้งค่าระยะเตือนภัย")
        self.resize(620, 430)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self.setWindowTitle("Third Eye - ตั้งค่าระยะเตือนภัย")
        self._load_values()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        header = QFrame(objectName="header")
        header.setFixedHeight(82)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(28, 12, 28, 12)
        header_layout.setSpacing(2)

        title = QLabel("ตั้งค่าระยะเตือนภัย")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        subtitle = QLabel("กำหนดระยะเป็นเมตร ระบบจะใช้ค่านี้ในการแบ่ง อันตราย / ระวัง / ปลอดภัย")
        subtitle.setStyleSheet("color:#94a3b8; font-size:13px;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        body = QFrame()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(26, 26, 26, 26)
        body_layout.setSpacing(18)

        card = QFrame(objectName="card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 26, 28, 26)
        card_layout.setSpacing(20)

        explanation = QLabel(
            "หลักการทำงาน\n"
            "• อันตราย: ระยะน้อยกว่าหรือเท่ากับค่าที่กำหนด\n"
            "• ระวัง: ระยะมากกว่าอันตราย แต่ไม่เกินค่าระวัง\n"
            "• ปลอดภัย: ระยะมากกว่าค่าระวัง"
        )
        explanation.setStyleSheet(
            "background:#0b1220; color:#cbd5e1; border-radius:12px; "
            "padding:14px; font-size:14px; line-height:1.4;"
        )
        card_layout.addWidget(explanation)

        self.danger_spin = self._make_spinbox()
        self.warning_spin = self._make_spinbox()
        card_layout.addLayout(self._make_row("ระยะอันตราย", "Danger", self.danger_spin, "#ef4444"))
        card_layout.addLayout(self._make_row("ระยะระวัง", "Warning", self.warning_spin, "#f59e0b"))

        self.preview = QLabel("")
        self.preview.setStyleSheet(
            "background:#101827; color:#e5e7eb; border:1px solid #243244; "
            "border-radius:12px; padding:12px; font-size:14px;"
        )
        card_layout.addWidget(self.preview)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#facc15; font-size:13px;")
        card_layout.addWidget(self.status)

        save_button = QPushButton("บันทึกการตั้งค่า")
        save_button.setObjectName("primaryButton")
        save_button.setFixedHeight(48)
        save_button.clicked.connect(self.save_values)
        card_layout.addWidget(save_button)

        body_layout.addWidget(card)
        main.addWidget(header)
        main.addWidget(body)

        self.danger_spin.valueChanged.connect(self._update_preview)
        self.warning_spin.valueChanged.connect(self._update_preview)

    @staticmethod
    def _make_spinbox():
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0.1, 200.0)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(0.5)
        spinbox.setSuffix(" m")
        spinbox.setFixedWidth(150)
        return spinbox

    @staticmethod
    def _make_row(title_text, tag_text, widget, color):
        row = QHBoxLayout()
        row.setSpacing(12)

        label_box = QVBoxLayout()
        title = QLabel(title_text)
        title.setFont(QFont("Arial", 14, QFont.Bold))
        tag = QLabel(tag_text)
        tag.setStyleSheet(f"color:{color}; font-size:12px; font-weight:700;")
        label_box.addWidget(title)
        label_box.addWidget(tag)

        row.addLayout(label_box)
        row.addStretch()
        row.addWidget(widget)
        return row

    def _load_values(self):
        thresholds = load_distance_thresholds()
        self.danger_spin.setValue(thresholds["danger"])
        self.warning_spin.setValue(thresholds["warning"])
        self._update_preview()

    # Keep all user-facing validation text in one place and ensure it remains
    # readable even when an older settings file contains legacy text.
    def _update_preview(self):
        danger = self.danger_spin.value()
        warning = self.warning_spin.value()
        if warning <= danger:
            self.preview.setText("⚠ ระยะเตือนต้องมากกว่าระยะอันตราย")
            self.preview.setStyleSheet(
                "background:#3b1d1d; color:#fecaca; border:1px solid #ef4444; "
                "border-radius:12px; padding:12px; font-size:14px;"
            )
            return
        self.preview.setStyleSheet(
            "background:#101827; color:#e5e7eb; border:1px solid #243244; "
            "border-radius:12px; padding:12px; font-size:14px;"
        )
        self.preview.setText(
            "ตัวอย่างการแบ่งระดับ\n"
            f"อันตราย: ไม่เกิน {danger:.1f} เมตร\n"
            f"ระวัง: มากกว่า {danger:.1f} ถึง {warning:.1f} เมตร\n"
            f"ปลอดภัย: มากกว่า {warning:.1f} เมตร"
        )

    def save_values(self):
        danger = self.danger_spin.value()
        warning = self.warning_spin.value()
        if warning <= danger:
            self.status.setStyleSheet("color:#f87171; font-size:13px; font-weight:700;")
            self.status.setText("กรุณาตั้งค่าระยะเตือนให้มากกว่าระยะอันตราย")
            return
        thresholds = save_distance_thresholds(danger, warning)
        self.status.setStyleSheet("color:#22c55e; font-size:13px; font-weight:700;")
        self.status.setText("บันทึกสำเร็จ ใช้ค่าระยะใหม่แล้ว")
        self.thresholds_saved.emit(thresholds["danger"], thresholds["warning"])
        self.close()
