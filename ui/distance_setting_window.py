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


class DistanceSettingWindow(QMainWindow):
    thresholds_saved = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Eye - Distance Setting")
        self.resize(520, 360)
        self._build_ui()
        self._load_values()

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
        title = QLabel("Distance Warning Setting")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title)

        body = QFrame()
        body.setStyleSheet("background:#6A6A6A;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(35, 25, 35, 25)
        body_layout.setSpacing(18)

        description = QLabel(
            "Set the distance thresholds in meters.\n"
            "Danger: distance <= danger value\n"
            "Warning: distance <= warning value\n"
            "Safe: distance > warning value"
        )
        description.setStyleSheet("color:white; font-size:14px;")
        body_layout.addWidget(description)

        self.danger_spin = self._make_spinbox()
        self.warning_spin = self._make_spinbox()

        body_layout.addLayout(self._make_row("Danger distance (meters)", self.danger_spin))
        body_layout.addLayout(self._make_row("Warning distance (meters)", self.warning_spin))

        self.status = QLabel("")
        self.status.setStyleSheet("color:yellow; font-size:13px;")
        body_layout.addWidget(self.status)
        body_layout.addStretch()

        save_button = QPushButton("Save")
        save_button.setFixedSize(160, 45)
        save_button.setStyleSheet("background:#E0E0E0; border-radius:10px; font-size:16px;")
        save_button.clicked.connect(self.save_values)
        body_layout.addWidget(save_button, alignment=Qt.AlignCenter)

        main.addWidget(header)
        main.addWidget(body)

    @staticmethod
    def _make_spinbox():
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0.1, 200.0)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(0.5)
        spinbox.setSuffix(" m")
        spinbox.setFixedWidth(140)
        spinbox.setStyleSheet("font-size:15px;")
        return spinbox

    @staticmethod
    def _make_row(label_text, widget):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("color:white; font-size:15px;")
        row.addWidget(label)
        row.addStretch()
        row.addWidget(widget)
        return row

    def _load_values(self):
        thresholds = load_distance_thresholds()
        self.danger_spin.setValue(thresholds["danger"])
        self.warning_spin.setValue(thresholds["warning"])

    def save_values(self):
        danger = self.danger_spin.value()
        warning = self.warning_spin.value()
        if warning <= danger:
            self.status.setText("Warning distance must be greater than danger distance.")
            return

        thresholds = save_distance_thresholds(danger, warning)
        self.status.setText(
            f"Saved: Danger <= {thresholds['danger']} m, "
            f"Warning <= {thresholds['warning']} m, Safe > {thresholds['warning']} m"
        )
        self.thresholds_saved.emit(thresholds["danger"], thresholds["warning"])
