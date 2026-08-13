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

import sys


def _show_startup_error(error):
    """Show a useful dependency/runtime error instead of a silent crash."""
    message = (
        "Third Eye เริ่มระบบไม่ได้\n\n"
        f"สาเหตุ: {error}\n\n"
        "ตรวจสอบว่า Python environment มี PyTorch, PyQt5, OpenCV "
        "และ Ultralytics ครบ และใช้ Python เวอร์ชันเดียวกับที่ติดตั้งแพ็กเกจไว้"
    )
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox

        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Third Eye - เริ่มระบบไม่สำเร็จ", message)
    except Exception:
        print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        # PyQt ships DLLs that can conflict with PyTorch if Qt is imported
        # first. Load PyTorch before any PyQt module so Torch wins conflicts.
        import torch  # noqa: F401
        from PyQt5.QtWidgets import QApplication
        from ui.main_window import MainWindow
    except Exception as error:
        sys.exit(_show_startup_error(error))

    app = QApplication(sys.argv)
    try:
        win = MainWindow()
        win.show()
    except Exception as error:
        sys.exit(_show_startup_error(error))
    sys.exit(app.exec_())
