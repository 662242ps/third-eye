from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vision.model_config import (
    import_model_file,
    list_available_models,
    load_model_settings,
    save_model_settings,
)


APP_STYLE = """
QMainWindow { background: #0f172a; }
QWidget { font-family: 'Leelawadee UI'; }
QLabel { color: #e5e7eb; }
QFrame#card {
    background: #172033;
    border: 1px solid #2d3a4f;
    border-radius: 18px;
}
QComboBox {
    background: #0b1220; color: #f8fafc;
    border: 1px solid #334155; border-radius: 10px;
    padding: 8px 10px; font-size: 15px; min-height: 24px;
}
QComboBox QAbstractItemView {
    background: #0b1220; color: #f8fafc;
    selection-background-color: #2563eb;
}
QPushButton {
    background: #334155; color: #f8fafc; border: none;
    border-radius: 12px; padding: 11px; font-size: 15px; font-weight: 700;
}
QPushButton:hover { background: #475569; }
QPushButton#primaryButton {
    background: #2563eb; color: white;
}
QPushButton#primaryButton:hover { background: #1d4ed8; }
"""


class ModelSettingWindow(QMainWindow):
    model_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye - ตั้งค่าโมเดล")
        self.resize(560, 400)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._reload_models()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)

        title = QLabel("เลือกโมเดลตรวจจับ (YOLO)")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        hint = QLabel(
            "เลือกไฟล์โมเดล (.pt) จากรายการที่มีอยู่ในโฟลเดอร์ models "
            "หรือกด \"เพิ่มโมเดลจากไฟล์ในเครื่อง\" เพื่อนำเข้าไฟล์ใหม่ "
            "ไม่จำเป็นต้องตั้งชื่อไฟล์ว่า best.pt"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8; font-size:13px;")

        card = QFrame(objectName="card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 22, 22, 22)
        card_layout.setSpacing(16)

        self.model_combo = QComboBox()
        card_layout.addWidget(self.model_combo)

        add_button = QPushButton("เพิ่มโมเดลจากไฟล์ในเครื่อง...")
        add_button.clicked.connect(self.add_model_file)
        card_layout.addWidget(add_button)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#facc15; font-size:13px;")
        card_layout.addWidget(self.status)

        save_button = QPushButton("ใช้งานโมเดลนี้")
        save_button.setObjectName("primaryButton")
        save_button.setFixedHeight(46)
        save_button.clicked.connect(self.save_selection)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(card)
        layout.addStretch()
        layout.addWidget(save_button)

    def _reload_models(self, select_relative=None):
        current_relative, _ = load_model_settings()
        target = select_relative or current_relative

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        models = list_available_models()
        selected_index = 0
        for index, item in enumerate(models):
            self.model_combo.addItem(item["label"], item["path"])
            if item["path"] == target:
                selected_index = index
        if models:
            self.model_combo.setCurrentIndex(selected_index)
        self.model_combo.blockSignals(False)

        if not models:
            self.status.setText(
                "ไม่พบไฟล์โมเดล (.pt) ในโฟลเดอร์ models กรุณาเพิ่มไฟล์โมเดลก่อน"
            )

    def add_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "เลือกไฟล์โมเดล", "", "YOLO Model (*.pt)"
        )
        if not path:
            return

        try:
            relative = import_model_file(path)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "ไม่สามารถเพิ่มโมเดลได้", str(error))
            return

        self._reload_models(select_relative=relative)
        self.status.setStyleSheet("color:#22c55e; font-size:13px; font-weight:700;")
        self.status.setText(f"เพิ่มโมเดล \"{relative}\" แล้ว")

    def save_selection(self):
        relative = self.model_combo.currentData()
        if not relative:
            self.status.setStyleSheet("color:#ef4444; font-size:13px; font-weight:700;")
            self.status.setText("กรุณาเลือกโมเดลก่อนบันทึก")
            return

        save_model_settings(relative)
        self.model_selected.emit(relative)
        self.close()
