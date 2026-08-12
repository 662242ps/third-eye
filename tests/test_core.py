import unittest

from vision.alert_config import _normalize
from vision.distance_config import _normalize_thresholds
from vision.voice_alert import _thai_distance, _thai_integer
from vision.yolo_thread import YoloThread


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


if __name__ == "__main__":
    unittest.main()
