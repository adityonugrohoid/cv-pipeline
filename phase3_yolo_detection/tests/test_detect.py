"""Tests for phase3_yolo_detection.detect and visualize."""

import numpy as np
import pytest

from phase3_yolo_detection.detect import Detection


class TestDetectionDataclass:
    def test_fields(self) -> None:
        d = Detection(
            class_name="arrow",
            confidence=0.95,
            bbox=(100, 200, 50, 60),
            center=(125, 230),
        )
        assert d.class_name == "arrow"
        assert d.confidence == 0.95
        assert d.bbox == (100, 200, 50, 60)
        assert d.center == (125, 230)

    def test_bbox_tuple_length(self) -> None:
        d = Detection(class_name="test", confidence=0.5, bbox=(0, 0, 10, 10), center=(5, 5))
        assert len(d.bbox) == 4
        assert len(d.center) == 2


class TestDetectWithModel:
    """These tests require trained weights and are skipped if not available."""

    @pytest.fixture
    def weights_path(self) -> str:
        from pathlib import Path
        path = Path("models/best.pt")
        if not path.exists():
            pytest.skip("No trained weights at models/best.pt — run training first")
        return str(path)

    def test_detect_returns_detections(self, weights_path: str) -> None:
        from phase3_yolo_detection.detect import detect

        # Create a simple test image with a dark circle (should trigger some detection)
        import cv2
        img = np.ones((640, 640, 3), dtype=np.uint8) * 240
        cv2.circle(img, (320, 320), 40, (20, 20, 20), 2)
        cv2.line(img, (280, 280), (360, 360), (20, 20, 20), 2)
        cv2.line(img, (280, 360), (360, 280), (20, 20, 20), 2)

        detections = detect(img, weights=weights_path, conf=0.1)
        assert isinstance(detections, list)
        for d in detections:
            assert isinstance(d, Detection)
            assert 0 <= d.confidence <= 1

    def test_detect_on_val_image(self, weights_path: str) -> None:
        from pathlib import Path
        from phase3_yolo_detection.detect import detect

        val_imgs = list(Path("data/yolo_dataset/val/images").glob("*.png"))
        if not val_imgs:
            pytest.skip("No validation images found")

        detections = detect(str(val_imgs[0]), weights=weights_path, conf=0.1)
        assert isinstance(detections, list)
        # Validation image should have at least one detection at low conf
        assert len(detections) > 0

    def test_visualize_runs(self, weights_path: str) -> None:
        from phase3_yolo_detection.detect import detect
        from phase3_yolo_detection.visualize import draw_detections

        img = np.ones((640, 640, 3), dtype=np.uint8) * 240
        detections = detect(img, weights=weights_path, conf=0.1)
        annotated = draw_detections(img, detections)
        assert annotated.shape == img.shape
