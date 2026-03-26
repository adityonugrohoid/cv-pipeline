"""Run YOLO inference on a single image and return structured detections."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected object."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    center: tuple[int, int]


def detect(
    image: np.ndarray | str | Path,
    weights: str | Path = "models/best.pt",
    conf: float = 0.25,
    iou: float = 0.45,
) -> list[Detection]:
    """Run inference on an image and return detections.

    Args:
        image: BGR numpy array or path to image file.
        weights: Path to trained model weights.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        List of Detection objects.
    """
    model = YOLO(str(weights))

    if isinstance(image, (str, Path)):
        source = str(image)
    else:
        source = image

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    detections: list[Detection] = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = model.names.get(cls_id, str(cls_id))
            confidence = float(boxes.conf[i])

            # xyxy to xywh
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            cx, cy = x + w // 2, y + h // 2

            detections.append(Detection(
                class_name=cls_name,
                confidence=confidence,
                bbox=(x, y, w, h),
                center=(cx, cy),
            ))

    logger.info("Detected %d objects", len(detections))
    return detections
