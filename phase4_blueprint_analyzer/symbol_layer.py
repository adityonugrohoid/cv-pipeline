"""Pipeline stage wrapping Phase 3 YOLO symbol detection."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = Path("models/best.pt")


def run(
    image: np.ndarray,
    weights: str | Path = DEFAULT_WEIGHTS,
    conf: float = 0.25,
) -> list[dict[str, Any]]:
    """Detect construction symbols in an image using YOLO.

    Args:
        image: BGR image.
        weights: Path to trained YOLO weights.
        conf: Confidence threshold.

    Returns:
        List of detection dictionaries with class, confidence, bbox, center.
    """
    weights = Path(weights)
    if not weights.exists():
        logger.warning("YOLO weights not found at %s — skipping symbol detection", weights)
        return []

    from phase3_yolo_detection.detect import detect

    detections = detect(image, weights=weights, conf=conf)
    logger.info("Symbol layer: detected %d symbols", len(detections))

    return [
        {
            "class": d.class_name,
            "confidence": round(d.confidence, 4),
            "bbox": list(d.bbox),
            "center": list(d.center),
        }
        for d in detections
    ]
