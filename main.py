# main.py
import os

# ---- Fix OpenMP / Qt / Torch conflict ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Cap at 4 worker threads even on many-core machines: YOLO11s is a small
# model and gains little past ~4 threads, while claiming more starves the
# UI thread and the rest of the system (this was the main source of the
# app "eating" the whole CPU).
CPU_THREADS = max(1, min(4, os.cpu_count() or 2))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# PyQt ships DLLs that can conflict with PyTorch if Qt is imported first.
# Load PyTorch before any PyQt module so Torch's runtime dependencies win.
import torch
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
