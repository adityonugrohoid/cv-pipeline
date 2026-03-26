"""Draw YOLO detections on an image with class labels and confidence scores."""

import cv2
import numpy as np

from .detect import Detection

# Color palette (BGR) — one per class
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "arrow": (0, 180, 0),              # green
    "dimension_line": (200, 0, 0),      # blue
    "circle_x": (0, 0, 200),           # red
    "door_swing": (180, 100, 50),      # teal
    "electrical_outlet": (0, 140, 255), # orange
}

DEFAULT_COLOR = (128, 128, 128)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes, class labels, and confidence scores on a copy of the image.

    Args:
        image: BGR image as numpy array.
        detections: List of Detection objects.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()

    for det in detections:
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        x, y, w, h = det.bbox

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 4, y), color, -1)

        # Label text
        cv2.putText(
            annotated, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Center dot
        cv2.circle(annotated, det.center, 3, color, -1)

    return annotated
