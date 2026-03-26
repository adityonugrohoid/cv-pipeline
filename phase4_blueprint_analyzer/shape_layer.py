"""Pipeline stage wrapping Phase 1 shape detection."""

import logging
from typing import Any

import numpy as np

from phase1_shape_detection.detector import detect_shapes

logger = logging.getLogger(__name__)


def run(image: np.ndarray) -> list[dict[str, Any]]:
    """Detect geometric shapes in an image.

    Args:
        image: BGR image.

    Returns:
        List of shape dictionaries with type, bbox, center, and properties.
    """
    shapes = detect_shapes(image)
    logger.info("Shape layer: detected %d shapes", len(shapes))

    results: list[dict[str, Any]] = []
    for s in shapes:
        results.append({
            "shape_type": s.shape_type,
            "bbox": list(s.bbox),
            "center": list(s.center),
            "properties": s.properties,
        })
    return results
