"""Tests for phase3_yolo_detection.dataset."""

import tempfile
from pathlib import Path

import cv2
import yaml
import pytest

from phase3_yolo_detection.dataset import (
    CLASS_NAMES,
    generate_dataset,
    _to_yolo_label,
)


class TestToYoloLabel:
    def test_center_of_image(self) -> None:
        label = _to_yolo_label((100, 100, 200, 200), 0, 400, 400)
        parts = label.split()
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.5, abs=0.01)
        assert float(parts[2]) == pytest.approx(0.5, abs=0.01)
        assert float(parts[3]) == pytest.approx(0.5, abs=0.01)
        assert float(parts[4]) == pytest.approx(0.5, abs=0.01)

    def test_normalized_values_in_range(self) -> None:
        label = _to_yolo_label((10, 20, 50, 80), 2, 640, 640)
        parts = label.split()
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0


class TestGenerateDataset:
    def test_generates_correct_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = generate_dataset(tmpdir, num_images=20, val_ratio=0.2)

            assert yaml_path.exists()
            train_imgs = list((Path(tmpdir) / "train" / "images").glob("*.png"))
            val_imgs = list((Path(tmpdir) / "val" / "images").glob("*.png"))
            assert len(train_imgs) + len(val_imgs) == 20
            assert len(val_imgs) >= 3  # ~20% of 20

    def test_labels_match_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(tmpdir, num_images=10)

            for split in ("train", "val"):
                imgs = list((Path(tmpdir) / split / "images").glob("*.png"))
                for img_path in imgs:
                    label_path = Path(tmpdir) / split / "labels" / f"{img_path.stem}.txt"
                    assert label_path.exists(), f"Missing label for {img_path}"

    def test_label_format_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(tmpdir, num_images=5)

            label_files = list((Path(tmpdir) / "train" / "labels").glob("*.txt"))
            label_files += list((Path(tmpdir) / "val" / "labels").glob("*.txt"))

            for label_path in label_files:
                with open(label_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        assert len(parts) == 5, f"Expected 5 fields, got {len(parts)}"
                        cls_id = int(parts[0])
                        assert 0 <= cls_id < len(CLASS_NAMES)
                        for v in parts[1:]:
                            assert 0.0 <= float(v) <= 1.0

    def test_data_yaml_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = generate_dataset(tmpdir, num_images=5)

            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            assert "path" in data
            assert "train" in data
            assert "val" in data
            assert data["nc"] == len(CLASS_NAMES)
            assert data["names"] == CLASS_NAMES

    def test_images_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(tmpdir, num_images=5)

            imgs = list((Path(tmpdir) / "train" / "images").glob("*.png"))
            for img_path in imgs:
                img = cv2.imread(str(img_path))
                assert img is not None
                assert img.shape == (640, 640, 3)

    def test_reproducible_with_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            generate_dataset(tmp1, num_images=5, seed=123)
            generate_dataset(tmp2, num_images=5, seed=123)

            imgs1 = sorted((Path(tmp1) / "train" / "images").glob("*.png"))
            imgs2 = sorted((Path(tmp2) / "train" / "images").glob("*.png"))
            # Same number of train images
            assert len(imgs1) == len(imgs2)
