"""Shared frame preparation for live, video, and image-test sources."""

import cv2
import numpy as np


def letterbox_with_meta(frame, size=(640, 640), color=(0, 0, 0)):
    """Return a padded frame together with its source-to-model transform."""
    target_w, target_h = int(size[0]), int(size[1])
    if frame is None or frame.size == 0:
        return (
            np.zeros((target_h, target_w, 3), dtype=np.uint8),
            {
                "scale": 1.0,
                "offset_x": 0,
                "offset_y": 0,
                "source_w": 0,
                "source_h": 0,
            },
        )

    source_h, source_w = frame.shape[:2]
    scale = min(target_w / source_w, target_h / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

    output = np.full(
        (target_h, target_w, 3), color, dtype=frame.dtype
    )
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    output[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return output, {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "source_w": source_w,
        "source_h": source_h,
    }


def letterbox(frame, size=(640, 640), color=(0, 0, 0)):
    """Resize without distorting geometry, then pad to an exact square."""
    return letterbox_with_meta(frame, size, color)[0]


def unletterbox_box(box, transform):
    """Map a model-space box back to the original displayed frame."""
    scale = max(float(transform["scale"]), 1e-9)
    source_w = int(transform["source_w"])
    source_h = int(transform["source_h"])
    offset_x = float(transform["offset_x"])
    offset_y = float(transform["offset_y"])

    def map_x(value):
        return max(0, min(source_w - 1, int(round((value - offset_x) / scale))))

    def map_y(value):
        return max(0, min(source_h - 1, int(round((value - offset_y) / scale))))

    x1, y1, x2, y2 = box
    return map_x(x1), map_y(y1), map_x(x2), map_y(y2)


def unletterbox_points(points, source_shape, size=(640, 640)):
    """Map model-space polygon points back to an original frame."""
    source_h, source_w = source_shape[:2]
    target_w, target_h = int(size[0]), int(size[1])
    scale = min(target_w / source_w, target_h / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    mapped = []
    for x, y in np.asarray(points).reshape(-1, 2):
        mapped.append(
            [
                max(0, min(source_w - 1, int(round((x - offset_x) / scale)))),
                max(0, min(source_h - 1, int(round((y - offset_y) / scale)))),
            ]
        )
    return np.asarray(mapped, dtype=np.int32)
