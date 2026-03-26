"""Fine-tune YOLOv8n on a custom synthetic dataset."""

import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def train(
    data_yaml: str | Path,
    epochs: int = 20,
    imgsz: int = 640,
    weights_dir: str | Path = "models",
    project: str = "runs/train",
    name: str = "yolo_blueprint",
) -> Path:
    """Fine-tune YOLOv8n on a custom dataset.

    Args:
        data_yaml: Path to data.yaml for the dataset.
        epochs: Number of training epochs.
        imgsz: Input image size.
        weights_dir: Directory to copy best weights into.
        project: Training output project directory.
        name: Training run name.

    Returns:
        Path to the best weights file.
    """
    data_yaml = Path(data_yaml)
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLOv8n pretrained on COCO")
    model = YOLO("yolov8n.pt")

    logger.info("Starting training: %d epochs, imgsz=%d", epochs, imgsz)
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # Ultralytics may nest save_dir differently from project/name.
    # Use the trainer's actual save_dir to find weights reliably.
    save_dir = Path(model.trainer.save_dir) if model.trainer else Path(project) / name
    best_src = save_dir / "weights" / "best.pt"
    last_src = save_dir / "weights" / "last.pt"
    best_dst = weights_dir / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        logger.info("Saved best weights → %s", best_dst)
    elif last_src.exists():
        shutil.copy2(last_src, best_dst)
        logger.info("Saved last weights (best not found) → %s", best_dst)
    else:
        logger.warning("No weights found in %s", save_dir / "weights")

    return best_dst
