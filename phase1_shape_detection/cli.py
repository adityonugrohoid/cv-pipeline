"""CLI entrypoint for Phase 1 shape detection.

Usage:
    python -m phase1_shape_detection.cli detect --input image.png --output output.png --json results.json
    python -m phase1_shape_detection.cli generate  # create sample test image
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from .annotator import annotate_image
from .detector import detect_shapes
from .export import export_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_image(output_path: str = "assets/sample_shapes.png") -> None:
    """Generate a sample test image with various geometric shapes."""
    import numpy as np

    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Rectangles (different colors)
    cv2.rectangle(img, (50, 50), (200, 150), (200, 50, 50), 3)       # blue rect
    cv2.rectangle(img, (600, 50), (750, 200), (50, 50, 200), 3)      # red rect

    # Squares
    cv2.rectangle(img, (300, 400), (420, 520), (50, 180, 50), 3)     # green square

    # Circles
    cv2.circle(img, (400, 130), 80, (180, 100, 50), 3)               # teal circle
    cv2.circle(img, (650, 420), 60, (120, 50, 180), 3)               # magenta circle

    # Triangles
    tri1 = np.array([[150, 350], [50, 520], [250, 520]], dtype=np.int32)
    cv2.drawContours(img, [tri1], -1, (0, 140, 255), 3)              # orange triangle

    tri2 = np.array([[550, 250], [480, 370], [620, 370]], dtype=np.int32)
    cv2.drawContours(img, [tri2], -1, (200, 0, 100), 3)              # purple triangle

    # Pentagon
    cx_p, cy_p, r_p = 200, 280, 50
    angles = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi / 2
    pentagon = np.array(
        [[int(cx_p + r_p * np.cos(a)), int(cy_p + r_p * np.sin(a))] for a in angles],
        dtype=np.int32,
    )
    cv2.drawContours(img, [pentagon], -1, (100, 180, 0), 3)          # green-blue pentagon

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    logger.info("Generated sample image with shapes → %s", path)


def cmd_detect(args: argparse.Namespace) -> None:
    """Run shape detection on an input image."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        sys.exit(1)

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error("Failed to read image: %s", input_path)
        sys.exit(1)

    shapes = detect_shapes(image)

    if args.output:
        annotated = annotate_image(image, shapes)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, annotated)
        logger.info("Saved annotated image → %s", args.output)

    if args.json:
        data = export_json(shapes, args.json)
        logger.info("Exported %d shapes to %s", data["summary"]["total"], args.json)

    for i, s in enumerate(shapes, 1):
        x, y, w, h = s.bbox
        logger.info("  %d. %s at (%d,%d) size=%dx%d center=(%d,%d)",
                     i, s.shape_type, x, y, w, h, *s.center)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phase1_shape_detection",
        description="Contour-based shape detection in images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # detect
    p_detect = sub.add_parser("detect", help="Detect shapes in an image")
    p_detect.add_argument("--input", required=True, help="Input image path")
    p_detect.add_argument("--output", help="Output annotated image path")
    p_detect.add_argument("--json", help="Output JSON results path")

    # generate
    p_gen = sub.add_parser("generate", help="Generate sample test image")
    p_gen.add_argument("--output", default="assets/sample_shapes.png",
                       help="Output path for generated image")

    args = parser.parse_args()

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "generate":
        generate_sample_image(args.output)


if __name__ == "__main__":
    main()
