"""Contour-based shape detection for geometric shapes in images.

Detects rectangles, circles, triangles, and polygons using OpenCV contour
analysis. Extends color-segmented detection from the reference implementation
to handle overlapping shapes drawn in different colors, with a fallback to
standard edge detection for monochrome images.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Shape:
    """A detected geometric shape with its properties."""

    shape_type: str
    contour: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    center: tuple[int, int]
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Color segmentation (from reference — handles overlapping colored shapes)
# ---------------------------------------------------------------------------

def _find_unique_colors(
    image: np.ndarray,
    bg_threshold: int = 240,
    color_distance: int = 40,
) -> list[np.ndarray]:
    """Discover distinct line colors in the image.

    Masks out near-white background, quantizes remaining pixels, then
    clusters nearby colors so JPEG artifacts don't create false groups.
    """
    mask = np.any(image < bg_threshold, axis=2)
    fg_pixels = image[mask].astype(np.float32)

    if len(fg_pixels) == 0:
        return []

    quantized = (fg_pixels // 8) * 8
    unique = np.unique(quantized.astype(np.uint8), axis=0)

    clusters: list[np.ndarray] = []
    for color in unique:
        merged = False
        for center in clusters:
            if np.linalg.norm(color.astype(float) - center.astype(float)) < color_distance:
                merged = True
                break
        if not merged:
            clusters.append(color)

    return clusters


def _color_mask(image: np.ndarray, color: np.ndarray, tolerance: int = 35) -> np.ndarray:
    """Create a binary mask isolating pixels near the given BGR color."""
    lower = np.clip(color.astype(int) - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(color.astype(int) + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(image, lower, upper)

    white_mask = cv2.inRange(image, (230, 230, 230), (255, 255, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))

    # Dilate to close gaps from anti-aliasing, then close to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Fill enclosed regions so outline shapes (e.g. unfilled circles) become
    # solid — this makes contour area and circularity meaningful.
    fill_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, fill_contours, -1, 255, cv2.FILLED)

    return mask


# ---------------------------------------------------------------------------
# Shape classification
# ---------------------------------------------------------------------------

def _classify_contour(
    contour: np.ndarray,
    epsilon_factor: float = 0.02,
    min_area: float = 500.0,
    circularity_threshold: float = 0.7,
) -> Shape | None:
    """Classify a single contour into a Shape or return None if too small."""
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return None

    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
    vertices = len(approx)

    x, y, w, h = cv2.boundingRect(approx)
    cx, cy = x + w // 2, y + h // 2

    # Circularity: 4*pi*area / perimeter^2  (1.0 = perfect circle)
    circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0

    if circularity > circularity_threshold and vertices >= 6:
        # Circle
        (circle_cx, circle_cy), radius = cv2.minEnclosingCircle(contour)
        circle_cx, circle_cy, radius = int(circle_cx), int(circle_cy), int(radius)
        return Shape(
            shape_type="circle",
            contour=contour,
            bbox=(x, y, w, h),
            center=(circle_cx, circle_cy),
            properties={
                "radius": radius,
                "area": float(area),
                "circularity": float(circularity),
            },
        )
    elif vertices == 3:
        # Triangle
        return Shape(
            shape_type="triangle",
            contour=contour,
            bbox=(x, y, w, h),
            center=(cx, cy),
            properties={
                "vertices": vertices,
                "width": w,
                "height": h,
                "area": float(area),
            },
        )
    elif vertices == 4:
        # Rectangle (or square)
        aspect_ratio = w / h if h > 0 else 0
        shape_type = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
        return Shape(
            shape_type=shape_type,
            contour=contour,
            bbox=(x, y, w, h),
            center=(cx, cy),
            properties={
                "width": w,
                "height": h,
                "area": float(area),
                "aspect_ratio": float(aspect_ratio),
            },
        )
    elif vertices >= 5:
        # Polygon
        return Shape(
            shape_type="polygon",
            contour=contour,
            bbox=(x, y, w, h),
            center=(cx, cy),
            properties={
                "vertices": vertices,
                "width": w,
                "height": h,
                "area": float(area),
            },
        )

    return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _is_duplicate(shape: Shape, existing: list[Shape], iou_threshold: float = 0.5) -> bool:
    """Check if a shape overlaps significantly with an already-detected one."""
    x1, y1, w1, h1 = shape.bbox
    for other in existing:
        x2, y2, w2, h2 = other.bbox
        xi = max(x1, x2)
        yi = max(y1, y2)
        xa = min(x1 + w1, x2 + w2)
        ya = min(y1 + h1, y2 + h2)
        inter = max(0, xa - xi) * max(0, ya - yi)
        union = w1 * h1 + w2 * h2 - inter
        if union > 0 and inter / union > iou_threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Main detection API
# ---------------------------------------------------------------------------

def detect_shapes(
    image: np.ndarray,
    epsilon_factor: float = 0.02,
    min_area: float = 500.0,
) -> list[Shape]:
    """Detect geometric shapes in an image.

    Uses per-color segmentation when distinct line colors are found,
    falling back to Canny edge detection for monochrome images.

    Args:
        image: BGR image as numpy array.
        epsilon_factor: Contour approximation accuracy (fraction of perimeter).
        min_area: Minimum contour area in pixels to consider.

    Returns:
        List of detected Shape objects.
    """
    colors = _find_unique_colors(image)
    logger.info("Found %d unique line color(s)", len(colors))

    all_shapes: list[Shape] = []

    if colors:
        for color in colors:
            mask = _color_mask(image, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                shape = _classify_contour(cnt, epsilon_factor, min_area)
                if shape and not _is_duplicate(shape, all_shapes):
                    all_shapes.append(shape)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            shape = _classify_contour(cnt, epsilon_factor, min_area)
            if shape and not _is_duplicate(shape, all_shapes):
                all_shapes.append(shape)

    logger.info("Detected %d shape(s): %s", len(all_shapes),
                ", ".join(f"{s.shape_type}" for s in all_shapes))
    return all_shapes
