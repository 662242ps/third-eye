# main.py
import os

# ---- Fix OpenMP / Qt / Torch conflict ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CPU_THREADS = max(1, min(8, os.cpu_count() or 2))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
