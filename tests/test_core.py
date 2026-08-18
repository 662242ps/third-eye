import unittest
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from vision.alert_config import _normalize
from vision.alert_sound import ALARM_PROFILES
from vision.camera_config import (
    _normalize as normalize_camera,
    estimate_focal_length,
)
from vision.distance_config import _normalize_thresholds
from vision.frame_utils import letterbox, letterbox_with_meta, unletterbox_box
from vision.object_height_config import DEFAULT_OBJECT_HEIGHTS, _normalize as normalize_object_heights
from vision.voice_alert import (
    SEGMENT_VOICE_ONLY,
    VOICE_SPEECH_ENABLED,
    VoiceAnnouncer,
    _prune_wav_cache,
    _thai_distance,
    _thai_integer,
)
from vision.voice_segments import (
    SEGMENT_LENGTH_SCALES,
    SEGMENT_TEXT,
    SEGMENT_TTS_TEXT,
    SEGMENT_TRIM_THRESHOLD,
    build_segment_keys,
    trim_pcm_frames,
)
from vision.yolo_thread import (
    CPU_INFERENCE_INTERVAL_S,
    MAX_TRACK_SPEED_PX_S,
    MODEL_IMGSZ,
    YoloThread,
)
from vision.model_config import (
    DEFAULT_MODEL_CONF,
    DEFAULT_MODEL_IOU,
    DEFAULT_MODEL_RELATIVE,
    normalize_model_thresholds,
)
from ui.main_window import DISPLAY_FRAME_SIZE, MODEL_FRAME_SIZE, _video_timer_interval


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

    def test_alert_volume_is_clamped(self):
        values = _normalize({"siren_volume": 140, "voice_volume": -10})
        self.assertEqual(values["siren_volume"], 100)
        self.assertEqual(values["voice_volume"], 0)

    def test_non_finite_calibration_values_use_defaults(self):
        self.assertEqual(
            normalize_camera({"focal_length": math.nan})["focal_length"],
            600.0,
        )
        self.assertEqual(
            _normalize_thresholds({"danger": math.inf, "warning": 40})["danger"],
            20.0,
        )

    def test_focal_length_calibration_formula(self):
        self.assertEqual(estimate_focal_length(20, 60, 1.5), 800.0)
        with self.assertRaises(ValueError):
            estimate_focal_length(0, 60, 1.5)

    def test_object_heights_are_complete_and_safe(self):
        values = normalize_object_heights({"car": 1.75, "truck": "bad"})
        self.assertEqual(values["car"], 1.75)
        self.assertEqual(values["truck"], DEFAULT_OBJECT_HEIGHTS["truck"])
        self.assertEqual(set(values), set(DEFAULT_OBJECT_HEIGHTS))

    def test_object_height_out_of_range_uses_default(self):
        values = normalize_object_heights({"car": 0, "person": 99})
        self.assertEqual(values["car"], DEFAULT_OBJECT_HEIGHTS["car"])
        self.assertEqual(values["person"], DEFAULT_OBJECT_HEIGHTS["person"])

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
        self.assertEqual(DEFAULT_MODEL_RELATIVE, "model5_150/best.pt")

    def test_model_thresholds_are_normalized(self):
        self.assertEqual(
            normalize_model_thresholds(1.5, -1),
            {"conf": DEFAULT_MODEL_CONF, "iou": DEFAULT_MODEL_IOU},
        )
        self.assertEqual(
            normalize_model_thresholds(0.42, 0.67),
            {"conf": 0.42, "iou": 0.67},
        )

    def test_cpu_inference_interval_is_50ms(self):
        self.assertEqual(CPU_INFERENCE_INTERVAL_S, 0.05)

    def test_model_box_maps_to_display_letterbox(self):
        # A 1920x1080 source becomes 640x640 for inference (140 px top/bottom
        # padding) and 640x360 for display. The same source box must be drawn
        # at the matching 640x360 coordinates.
        transform = {
            "model": {
                "scale": 640 / 1920,
                "offset_x": 0,
                "offset_y": 140,
            },
            "display": {
                "scale": 640 / 1920,
                "offset_x": 0,
                "offset_y": 0,
            },
        }
        self.assertEqual(
            YoloThread._map_model_box_to_display(
                (100, 200, 300, 400), transform
            ),
            (100, 60, 300, 260),
        )

    def test_video_display_interval_is_bounded(self):
        self.assertEqual(_video_timer_interval(60), 33)
        self.assertEqual(_video_timer_interval(10), 100)
        self.assertEqual(_video_timer_interval(float("nan")), 33)

    def test_display_frame_size_is_lightweight_and_widescreen(self):
        self.assertEqual(DISPLAY_FRAME_SIZE, (640, 360))
        self.assertEqual(MODEL_FRAME_SIZE, (640, 640))

    def test_prediction_velocity_is_bounded(self):
        predicted = YoloThread._predict_box(
            {
                "box": (100, 100, 200, 200),
                "timestamp": 0.0,
                "velocity": (MAX_TRACK_SPEED_PX_S * 100, -MAX_TRACK_SPEED_PX_S * 100),
            },
            0.4,
        )
        self.assertLessEqual(predicted[0], 1100)
        self.assertGreaterEqual(predicted[1], -900)

    def test_warning_siren_is_quieter_and_slower_than_danger(self):
        warning = ALARM_PROFILES["WARNING"]
        danger = ALARM_PROFILES["DANGER"]
        self.assertLess(warning["volume"], danger["volume"])
        self.assertLess(warning["frequency_hz"][1], danger["frequency_hz"][1])
        warning_cycle = warning["tone_ms"] + warning["pause_ms"]
        danger_cycle = danger["tone_ms"] + danger["pause_ms"]
        self.assertGreater(warning_cycle, danger_cycle)

    def test_voice_segments_cover_dynamic_alert(self):
        keys = build_segment_keys("car", "DANGER", 5.8)
        self.assertEqual(
            keys,
            ["danger", "has", "car", "in_range", "number_05", "meter"],
        )
        self.assertEqual(
            len([key for key in SEGMENT_TEXT if key.startswith("number_")]),
            100,
        )

    def test_voice_segments_trim_model_padding(self):
        import struct

        silence = b"\0" * 20
        speech = b"".join(struct.pack("<h", value) for value in (1000, -1200, 800))
        trimmed = trim_pcm_frames(silence + speech + silence, 2, 1)
        self.assertEqual(trimmed, speech)

    def test_meter_voice_keeps_normal_speed_and_quiet_tail(self):
        self.assertEqual(SEGMENT_LENGTH_SCALES["meter"], 1.0)
        self.assertLess(SEGMENT_TRIM_THRESHOLD, 256)
        self.assertEqual(SEGMENT_TEXT["meter"], "\u0e40\u0e21\u0e15\u0e23   ")
        self.assertEqual(SEGMENT_TTS_TEXT["meter"], "\u0e40\u0e21\u0e49\u0e15")

    def test_only_male_voice_is_supported(self):
        self.assertEqual(
            _normalize({"voice_model": "removed_voice"})["voice_model"],
            "th_m_1",
        )

    def test_runtime_voice_does_not_load_tts_model(self):
        announcer = VoiceAnnouncer()
        self.assertTrue(VOICE_SPEECH_ENABLED)
        self.assertTrue(SEGMENT_VOICE_ONLY)
        self.assertTrue(announcer.available)
        self.assertFalse(announcer._bundled_tts_available())
        self.assertFalse(announcer._play_bundled_tts("ทดสอบ"))

    def test_runtime_voice_cache_is_bounded(self):
        with TemporaryDirectory() as directory:
            for index in range(5):
                Path(directory, f"alert_{index}.wav").write_bytes(b"RIFF")
            _prune_wav_cache(directory, 2)
            self.assertLessEqual(
                len(list(Path(directory).glob("*.wav"))), 2
            )

    def test_letterbox_box_maps_back_to_original_frame(self):
        frame = __import__("numpy").zeros((360, 640, 3), dtype="uint8")
        _, transform = letterbox_with_meta(frame)
        self.assertEqual(
            unletterbox_box((160, 230, 480, 410), transform),
            (160, 90, 480, 270),
        )


if __name__ == "__main__":
    unittest.main()
