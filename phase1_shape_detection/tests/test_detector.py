"""Tests for phase1_shape_detection.detector."""

from collections import Counter

import cv2
import numpy as np
import pytest

from phase1_shape_detection.detector import Shape, detect_shapes


def _make_white_image(width: int = 800, height: int = 600) -> np.ndarray:
    return np.ones((height, width, 3), dtype=np.uint8) * 255


class TestDetectShapes:
    """Integration tests using programmatically generated images."""

    def test_detects_rectangle(self) -> None:
        img = _make_white_image()
        cv2.rectangle(img, (100, 100), (300, 200), (200, 50, 50), 3)
        shapes = detect_shapes(img)
        rects = [s for s in shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 1
        assert rects[0].properties["width"] > 0
        assert rects[0].properties["height"] > 0

    def test_detects_circle(self) -> None:
        img = _make_white_image()
        cv2.circle(img, (400, 300), 100, (180, 100, 50), 3)
        shapes = detect_shapes(img)
        circles = [s for s in shapes if s.shape_type == "circle"]
        assert len(circles) == 1
        assert circles[0].properties["radius"] > 80

    def test_detects_triangle(self) -> None:
        img = _make_white_image()
        tri = np.array([[400, 100], [200, 400], [600, 400]], dtype=np.int32)
        cv2.drawContours(img, [tri], -1, (0, 140, 255), 3)
        shapes = detect_shapes(img)
        triangles = [s for s in shapes if s.shape_type == "triangle"]
        assert len(triangles) == 1

    def test_detects_polygon(self) -> None:
        img = _make_white_image()
        cx, cy, r = 400, 300, 100
        angles = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi / 2
        pentagon = np.array(
            [[int(cx + r * np.cos(a)), int(cy + r * np.sin(a))] for a in angles],
            dtype=np.int32,
        )
        cv2.drawContours(img, [pentagon], -1, (100, 180, 0), 3)
        shapes = detect_shapes(img)
        polys = [s for s in shapes if s.shape_type == "polygon"]
        assert len(polys) == 1
        assert polys[0].properties["vertices"] == 5

    def test_sample_image_detects_all_shapes(self) -> None:
        """Run detection on the full sample image and verify totals."""
        img = cv2.imread("assets/sample_shapes.png")
        assert img is not None, "assets/sample_shapes.png must exist"
        shapes = detect_shapes(img)
        counts = Counter(s.shape_type for s in shapes)

        assert len(shapes) == 8
        assert counts["triangle"] == 2
        assert counts["circle"] == 2
        assert counts["rectangle"] == 1
        assert counts["square"] == 2
        assert counts["polygon"] == 1

    def test_ignores_small_contours(self) -> None:
        img = _make_white_image()
        cv2.rectangle(img, (100, 100), (110, 110), (200, 50, 50), 1)  # tiny
        shapes = detect_shapes(img, min_area=500)
        assert len(shapes) == 0

    def test_shape_has_required_fields(self) -> None:
        img = _make_white_image()
        cv2.rectangle(img, (100, 100), (300, 250), (200, 50, 50), 3)
        shapes = detect_shapes(img)
        assert len(shapes) >= 1
        s = shapes[0]
        assert isinstance(s, Shape)
        assert s.shape_type in ("rectangle", "square", "circle", "triangle", "polygon")
        assert len(s.bbox) == 4
        assert len(s.center) == 2
        assert isinstance(s.contour, np.ndarray)
        assert isinstance(s.confidence, float)

    def test_empty_image_returns_no_shapes(self) -> None:
        img = _make_white_image()
        shapes = detect_shapes(img)
        assert shapes == []

    def test_fallback_edge_detection(self) -> None:
        """When no distinct colors exist, fallback to Canny edge detection."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), (30, 30, 30), 3)
        shapes = detect_shapes(img)
        assert len(shapes) >= 1
