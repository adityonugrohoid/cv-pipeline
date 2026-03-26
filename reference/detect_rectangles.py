"""Detect rectangular contours in an image and mark their centers with red dots.

Handles both isolated and overlapping rectangles by using color-based
segmentation — each unique line color is isolated into its own binary mask
so overlapping edges never merge.

Usage:
    python scripts/detect_rectangles.py [--input INPUT] [--output OUTPUT] [--generate]

Flags:
    --generate   Create a sample_test.jpg with rectangles before detection.
    --input      Path to input image (default: sample_test.jpg)
    --output     Path to output image (default: output.jpg)
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# BGR colors used to draw sample rectangles — kept here so detection
# can work on arbitrary images too (it auto-discovers colors).
SAMPLE_RECTANGLES = [
    ((100, 80),  (300, 220), (200, 160, 50)),   # teal
    ((400, 50),  (650, 250), (50, 180, 50)),     # green
    ((150, 320), (450, 500), (50, 50, 200)),     # red
    ((500, 300), (750, 520), (180, 100, 50)),    # blue
    ((320, 150), (520, 280), (120, 50, 180)),    # magenta
]


def generate_sample_image(path: str, width: int = 800, height: int = 600) -> np.ndarray:
    """Create a white image with multiple coloured rectangles (some overlapping)."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for pt1, pt2, color in SAMPLE_RECTANGLES:
        cv2.rectangle(img, pt1, pt2, color, thickness=3)

    cv2.imwrite(path, img)
    logger.info("Generated sample image with %d rectangles → %s", len(SAMPLE_RECTANGLES), path)
    return img


# ---------------------------------------------------------------------------
# Color-segmented detection (handles overlapping rectangles)
# ---------------------------------------------------------------------------

def _find_unique_colors(
    image: np.ndarray,
    bg_threshold: int = 240,
    color_distance: int = 40,
) -> list[np.ndarray]:
    """Discover distinct line colors via quantization and clustering.

    Handles JPEG artifacts by:
    1. Masking out near-white background pixels.
    2. Quantizing remaining pixel colors to reduce noise.
    3. Clustering nearby quantized colors within ``color_distance``.
    """
    # Mask: keep only clearly non-white pixels
    mask = np.any(image < bg_threshold, axis=2)
    fg_pixels = image[mask].astype(np.float32)

    if len(fg_pixels) == 0:
        return []

    # Quantize to reduce JPEG noise (32 levels per channel)
    quantized = (fg_pixels // 8) * 8
    unique = np.unique(quantized.astype(np.uint8), axis=0)

    # Greedy clustering: merge colors within Euclidean distance.
    # Keep centers fixed (no running average) to prevent drift.
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


def _detect_rects_in_mask(mask: np.ndarray, epsilon_factor: float = 0.02) -> list[dict]:
    """Find 4-vertex contours in a binary mask and return rectangle info."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[dict] = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 500:
            x, y, w, h = cv2.boundingRect(approx)
            cx, cy = x + w // 2, y + h // 2
            results.append({
                "contour": approx,
                "center": (cx, cy),
                "bounding_rect": (x, y, w, h),
            })
    return results


def _is_duplicate(rect: dict, existing: list[dict], iou_threshold: float = 0.5) -> bool:
    """Check if a rectangle overlaps significantly with any already-detected one."""
    x1, y1, w1, h1 = rect["bounding_rect"]
    for other in existing:
        x2, y2, w2, h2 = other["bounding_rect"]
        xi = max(x1, x2)
        yi = max(y1, y2)
        xa = min(x1 + w1, x2 + w2)
        ya = min(y1 + h1, y2 + h2)
        inter = max(0, xa - xi) * max(0, ya - yi)
        union = w1 * h1 + w2 * h2 - inter
        if union > 0 and inter / union > iou_threshold:
            return True
    return False


def detect_rectangles(image: np.ndarray, epsilon_factor: float = 0.02) -> list[dict]:
    """Detect rectangular contours using per-color segmentation.

    1. Discover unique non-background colors in the image.
    2. For each color, create a binary mask and find rectangle contours.
    3. Deduplicate near-identical detections via IoU.

    Falls back to standard edge detection if no distinct colors are found.
    """
    colors = _find_unique_colors(image)
    logger.info("Found %d unique line color(s)", len(colors))

    all_rects: list[dict] = []

    if colors:
        tolerance = 35
        for color in colors:
            lower = np.clip(color.astype(int) - tolerance, 0, 255).astype(np.uint8)
            upper = np.clip(color.astype(int) + tolerance, 0, 255).astype(np.uint8)
            mask = cv2.inRange(image, lower, upper)
            # Also exclude near-white pixels that might fall in range
            white_mask = cv2.inRange(image, (230, 230, 230), (255, 255, 255))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))
            # Dilate to close gaps at corners / JPEG fringes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)
            rects = _detect_rects_in_mask(mask, epsilon_factor)
            for r in rects:
                if not _is_duplicate(r, all_rects):
                    all_rects.append(r)
    else:
        # Fallback: plain edge detection for images without distinct colors
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        all_rects = _detect_rects_in_mask(edges, epsilon_factor)

    logger.info("Detected %d rectangle(s)", len(all_rects))
    return all_rects


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_image(image: np.ndarray, rectangles: list[dict]) -> np.ndarray:
    """Draw green contours and red center dots on a copy of the image."""
    annotated = image.copy()

    for rect in rectangles:
        cv2.drawContours(annotated, [rect["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(annotated, rect["center"], 6, (0, 0, 255), -1)

        cx, cy = rect["center"]
        label = f"({cx}, {cy})"
        cv2.putText(annotated, label, (cx + 10, cy - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return annotated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect rectangles in an image.")
    parser.add_argument("--input", default="sample_test.jpg", help="Input image path")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    parser.add_argument("--generate", action="store_true", help="Generate sample_test.jpg first")
    args = parser.parse_args()

    if args.generate:
        generate_sample_image(args.input)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input image not found: %s (use --generate to create one)", input_path)
        return

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error("Failed to read image: %s", input_path)
        return

    rectangles = detect_rectangles(image)
    annotated = annotate_image(image, rectangles)

    cv2.imwrite(args.output, annotated)
    logger.info("Saved annotated image → %s", args.output)

    for i, rect in enumerate(rectangles, 1):
        x, y, w, h = rect["bounding_rect"]
        cx, cy = rect["center"]
        logger.info("  Rectangle %d: pos=(%d,%d) size=%dx%d center=(%d,%d)", i, x, y, w, h, cx, cy)


if __name__ == "__main__":
    main()
