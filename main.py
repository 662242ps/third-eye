# main.py
import os

# ---- Fix OpenMP / Qt / Torch conflict ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Keep up to four YOLO workers, while reserving one logical CPU for Qt,
# decoding, and audio on smaller machines.
CPU_THREADS = max(1, min(4, (os.cpu_count() or 2) - 1))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
# Some older Windows sessions keep this deprecated variable globally. Remove
# it before Qt starts so it cannot emit a warning on every application launch.
os.environ.pop("QT_DEVICE_PIXEL_RATIO", None)

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
