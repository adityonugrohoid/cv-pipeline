"""Evaluate a trained YOLO model: mAP, per-class precision/recall, confusion matrix."""

import logging
from pathlib import Path
from typing import Any

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def evaluate(
    weights: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    save_confusion_matrix: bool = True,
    project: str = "runs/evaluate",
    name: str = "val",
) -> dict[str, Any]:
    """Run validation on a trained model and report metrics.

    Args:
        weights: Path to model weights (.pt file).
        data_yaml: Path to data.yaml.
        imgsz: Input image size.
        conf: Confidence threshold for predictions.
        save_confusion_matrix: Whether to save the confusion matrix plot.
        project: Output project directory.
        name: Run name.

    Returns:
        Dictionary with mAP, per-class metrics, and confusion matrix path.
    """
    weights = Path(weights)
    data_yaml = Path(data_yaml)

    logger.info("Loading model from %s", weights)
    model = YOLO(str(weights))

    logger.info("Running validation on %s", data_yaml)
    results = model.val(
        data=str(data_yaml.resolve()),
        imgsz=imgsz,
        conf=conf,
        project=project,
        name=name,
        exist_ok=True,
        plots=save_confusion_matrix,
    )

    # Extract metrics
    metrics: dict[str, Any] = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    # Per-class metrics
    class_names = model.names
    per_class: list[dict[str, Any]] = []
    if results.box.ap_class_index is not None:
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_idx = int(cls_idx)
            per_class.append({
                "class": class_names.get(cls_idx, str(cls_idx)),
                "precision": float(results.box.p[i]),
                "recall": float(results.box.r[i]),
                "ap50": float(results.box.ap50[i]),
            })
    metrics["per_class"] = per_class

    # Confusion matrix path
    cm_path = Path(project) / name / "confusion_matrix.png"
    metrics["confusion_matrix"] = str(cm_path) if cm_path.exists() else None

    logger.info("mAP@50: %.3f, mAP@50-95: %.3f", metrics["mAP50"], metrics["mAP50_95"])
    for cls in per_class:
        logger.info("  %s: P=%.3f R=%.3f AP50=%.3f",
                     cls["class"], cls["precision"], cls["recall"], cls["ap50"])

    return metrics
