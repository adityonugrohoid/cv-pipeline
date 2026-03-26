"""Generate a synthetic dataset of construction-style symbols for YOLO training.

Draws arrows, dimension lines, circles with X (no-entry), door swings, and
electrical outlets on random backgrounds. Outputs YOLO-format labels and a
data.yaml for ultralytics.
"""

import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "arrow",
    "dimension_line",
    "circle_x",
    "door_swing",
    "electrical_outlet",
]

CLASS_MAP: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Symbol drawing functions — each returns (x, y, w, h) in pixel coords
# ---------------------------------------------------------------------------

def _draw_arrow(img: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    """Draw a straight arrow pointing in a random direction."""
    h, w = img.shape[:2]
    cx = rng.randint(60, w - 60)
    cy = rng.randint(60, h - 60)
    length = rng.randint(30, 60)
    angle = rng.uniform(0, 2 * np.pi)

    x1 = int(cx - length * np.cos(angle))
    y1 = int(cy - length * np.sin(angle))
    x2 = int(cx + length * np.cos(angle))
    y2 = int(cy + length * np.sin(angle))

    color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
    thickness = rng.randint(2, 3)
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, thickness, tipLength=0.3)

    pad = 10
    bx = max(min(x1, x2) - pad, 0)
    by = max(min(y1, y2) - pad, 0)
    bw = min(abs(x2 - x1) + 2 * pad, w - bx)
    bh = min(abs(y2 - y1) + 2 * pad, h - by)
    bw = max(bw, 20)
    bh = max(bh, 20)
    return bx, by, bw, bh


def _draw_dimension_line(img: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    """Draw a dimension line with end ticks and a measurement label."""
    h, w = img.shape[:2]
    horizontal = rng.choice([True, False])
    color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
    thickness = 2

    if horizontal:
        y = rng.randint(40, h - 40)
        x1 = rng.randint(30, w // 2 - 30)
        x2 = x1 + rng.randint(60, 150)
        x2 = min(x2, w - 30)
        cv2.line(img, (x1, y), (x2, y), color, thickness)
        tick = 8
        cv2.line(img, (x1, y - tick), (x1, y + tick), color, thickness)
        cv2.line(img, (x2, y - tick), (x2, y + tick), color, thickness)
        label = f"{rng.randint(5, 30)}'"
        cv2.putText(img, label, ((x1 + x2) // 2 - 10, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        return x1, y - tick - 12, x2 - x1, 2 * tick + 16
    else:
        x = rng.randint(40, w - 40)
        y1 = rng.randint(30, h // 2 - 30)
        y2 = y1 + rng.randint(60, 150)
        y2 = min(y2, h - 30)
        cv2.line(img, (x, y1), (x, y2), color, thickness)
        tick = 8
        cv2.line(img, (x - tick, y1), (x + tick, y1), color, thickness)
        cv2.line(img, (x - tick, y2), (x + tick, y2), color, thickness)
        label = f"{rng.randint(5, 30)}'"
        cv2.putText(img, label, (x + 6, (y1 + y2) // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        return x - tick - 4, y1, 2 * tick + 30, y2 - y1


def _draw_circle_x(img: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    """Draw a circle with an X through it (no-entry / prohibition symbol)."""
    h, w = img.shape[:2]
    cx = rng.randint(40, w - 40)
    cy = rng.randint(40, h - 40)
    r = rng.randint(15, 30)
    color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
    thickness = 2

    cv2.circle(img, (cx, cy), r, color, thickness)
    offset = int(r * 0.7)
    cv2.line(img, (cx - offset, cy - offset), (cx + offset, cy + offset), color, thickness)
    cv2.line(img, (cx - offset, cy + offset), (cx + offset, cy - offset), color, thickness)

    return cx - r, cy - r, 2 * r, 2 * r


def _draw_door_swing(img: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    """Draw a quarter-circle arc representing a door swing."""
    h, w = img.shape[:2]
    cx = rng.randint(50, w - 50)
    cy = rng.randint(50, h - 50)
    r = rng.randint(25, 45)
    color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
    thickness = 2

    start_angle = rng.choice([0, 90, 180, 270])
    cv2.ellipse(img, (cx, cy), (r, r), 0, start_angle, start_angle + 90, color, thickness)
    # Draw the wall line
    a1 = np.radians(start_angle)
    a2 = np.radians(start_angle + 90)
    cv2.line(img, (cx, cy), (int(cx + r * np.cos(a1)), int(cy + r * np.sin(a1))), color, thickness)
    cv2.line(img, (cx, cy), (int(cx + r * np.cos(a2)), int(cy + r * np.sin(a2))), color, thickness)

    return max(cx - r, 0), max(cy - r, 0), min(2 * r, w - max(cx - r, 0)), min(2 * r, h - max(cy - r, 0))


def _draw_electrical_outlet(img: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    """Draw a simple electrical outlet symbol (circle with two vertical slots)."""
    h, w = img.shape[:2]
    cx = rng.randint(30, w - 30)
    cy = rng.randint(30, h - 30)
    r = rng.randint(12, 22)
    color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
    thickness = 2

    cv2.circle(img, (cx, cy), r, color, thickness)
    slot_h = max(r // 3, 3)
    gap = max(r // 3, 4)
    cv2.line(img, (cx - gap, cy - slot_h), (cx - gap, cy + slot_h), color, thickness)
    cv2.line(img, (cx + gap, cy - slot_h), (cx + gap, cy + slot_h), color, thickness)

    return cx - r, cy - r, 2 * r, 2 * r


DRAW_FUNCTIONS = {
    "arrow": _draw_arrow,
    "dimension_line": _draw_dimension_line,
    "circle_x": _draw_circle_x,
    "door_swing": _draw_door_swing,
    "electrical_outlet": _draw_electrical_outlet,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _random_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    """Generate a slightly noisy near-white background."""
    base = rng.randint(230, 255)
    img = np.full((height, width, 3), base, dtype=np.uint8)
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _to_yolo_label(
    bbox: tuple[int, int, int, int],
    class_id: int,
    img_w: int,
    img_h: int,
) -> str:
    """Convert pixel bbox (x, y, w, h) to YOLO normalized format."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def generate_dataset(
    output_dir: str | Path = "data/yolo_dataset",
    num_images: int = 200,
    img_size: tuple[int, int] = (640, 640),
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Generate a synthetic YOLO-format dataset.

    Args:
        output_dir: Root directory for the dataset.
        num_images: Total number of images to generate.
        img_size: (width, height) of each image.
        val_ratio: Fraction of images for validation.
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated data.yaml file.
    """
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    # Create directory structure
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    num_val = int(num_images * val_ratio)
    num_train = num_images - num_val

    splits = (["train"] * num_train) + (["val"] * num_val)
    rng.shuffle(splits)

    width, height = img_size
    class_names = list(DRAW_FUNCTIONS.keys())

    for idx, split in enumerate(splits):
        img = _random_background(width, height, rng)
        labels: list[str] = []

        # Place 2-5 symbols per image
        n_symbols = rng.randint(2, 5)
        for _ in range(n_symbols):
            cls_name = rng.choice(class_names)
            cls_id = CLASS_MAP[cls_name]
            draw_fn = DRAW_FUNCTIONS[cls_name]
            bbox = draw_fn(img, rng)
            labels.append(_to_yolo_label(bbox, cls_id, width, height))

        fname = f"img_{idx:04d}"
        cv2.imwrite(str(output_dir / split / "images" / f"{fname}.png"), img)
        with open(output_dir / split / "labels" / f"{fname}.txt", "w") as f:
            f.write("\n".join(labels) + "\n")

    # Write data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    train_count = splits.count("train")
    val_count = splits.count("val")
    logger.info("Generated dataset: %d train, %d val images → %s",
                train_count, val_count, output_dir)
    return yaml_path
