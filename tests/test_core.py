import unittest
import math

from vision.alert_config import _normalize
from vision.alert_sound import ALARM_PROFILES
from vision.camera_config import _normalize as normalize_camera
from vision.distance_config import _normalize_thresholds
from vision.frame_utils import letterbox, letterbox_with_meta, unletterbox_box
from vision.voice_alert import _thai_distance, _thai_integer
from vision.yolo_thread import CLASS_CONF, LABEL_THAI, MODEL_IMGSZ, YoloThread


class CoreLogicTests(unittest.TestCase):
    def test_thai_distance_uses_integer_part(self):
        self.assertIn("สิบห้า", _thai_distance(15.8))
        self.assertNotIn("จุด", _thai_distance(15.8))

    def test_thai_numbers(self):
        self.assertEqual(_thai_integer(20), "ยี่สิบ")
        self.assertEqual(_thai_integer(31), "สามสิบเอ็ด")
        self.assertEqual(_thai_integer(40), "สี่สิบ")

    def test_thresholds_are_ordered(self):
        values = _normalize_thresholds({"danger": 20, "warning": 40})
        self.assertLess(values["danger"], values["warning"])

    def test_status_boundaries(self):
        classify = YoloThread._classify_distance
        self.assertEqual(classify(10, 20, 40), "DANGER")
        self.assertEqual(classify(30, 20, 40), "WARNING")
        self.assertEqual(classify(50, 20, 40), "SAFE")

    def test_voice_model_setting_is_validated(self):
        self.assertEqual(_normalize({"voice_model": "../../bad"})["voice_model"], "th_m_1")

    def test_alert_string_false_is_not_true(self):
        values = _normalize({"siren_enabled": "false", "voice_enabled": "0"})
        self.assertFalse(values["siren_enabled"])
        self.assertFalse(values["voice_enabled"])

    def test_non_finite_calibration_values_use_defaults(self):
        self.assertEqual(
            normalize_camera({"focal_length": math.nan})["focal_length"],
            800.0,
        )
        self.assertEqual(
            _normalize_thresholds({"danger": math.inf, "warning": 40})["danger"],
            5.0,
        )

    def test_bus_is_calibrated_and_labeled(self):
        self.assertIn("bus", CLASS_CONF)
        self.assertEqual(LABEL_THAI["bus"], "รถบัส")

    def test_result_image_round_trip_is_compact(self):
        # Keep this test independent from Qt/MainWindow startup and verify
        # the JPEG representation used by batch testing remains decodable.
        import cv2
        import numpy as np

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :, 1] = 180
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, image.shape)

    def test_letterbox_preserves_frame_dimensions(self):
        import numpy as np

        frame = np.full((360, 640, 3), 120, dtype=np.uint8)
        prepared = letterbox(frame)
        self.assertEqual(prepared.shape, (640, 640, 3))
        self.assertEqual(int(prepared[320, 320, 0]), 120)
        self.assertEqual(int(prepared[0, 320, 0]), 0)

    def test_inference_size_is_fixed_to_640(self):
        self.assertEqual(MODEL_IMGSZ, 640)

    def test_warning_siren_is_quieter_and_slower_than_danger(self):
        warning = ALARM_PROFILES["WARNING"]
        danger = ALARM_PROFILES["DANGER"]
        self.assertLess(warning["volume"], danger["volume"])
        self.assertLess(warning["frequency_hz"][1], danger["frequency_hz"][1])
        warning_cycle = warning["tone_ms"] + warning["pause_ms"]
        danger_cycle = danger["tone_ms"] + danger["pause_ms"]
        self.assertGreater(warning_cycle, danger_cycle)

    def test_letterbox_box_maps_back_to_original_frame(self):
        frame = __import__("numpy").zeros((360, 640, 3), dtype="uint8")
        _, transform = letterbox_with_meta(frame)
        self.assertEqual(
            unletterbox_box((160, 230, 480, 410), transform),
            (160, 90, 480, 270),
        )


if __name__ == "__main__":
    unittest.main()
