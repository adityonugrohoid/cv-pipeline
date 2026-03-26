"""CLI entrypoint for Phase 3 YOLO object detection.

Usage:
    python -m phase3_yolo_detection.cli generate --output data/yolo_dataset --num-images 200
    python -m phase3_yolo_detection.cli train --data data/yolo_dataset/data.yaml --epochs 20
    python -m phase3_yolo_detection.cli evaluate --weights models/best.pt --data data/yolo_dataset/data.yaml
    python -m phase3_yolo_detection.cli detect --input image.png --output output.png --weights models/best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a synthetic training dataset."""
    from .dataset import generate_dataset

    yaml_path = generate_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        seed=args.seed,
    )
    logger.info("Dataset ready. data.yaml → %s", yaml_path)


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune YOLOv8n on the dataset."""
    from .train import train

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("data.yaml not found: %s", data_path)
        sys.exit(1)

    weights_path = train(
        data_yaml=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
    )
    logger.info("Training complete. Weights → %s", weights_path)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model."""
    from .evaluate import evaluate

    weights_path = Path(args.weights)
    data_path = Path(args.data)
    if not weights_path.exists():
        logger.error("Weights not found: %s", weights_path)
        sys.exit(1)

    metrics = evaluate(
        weights=weights_path,
        data_yaml=data_path,
    )

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics exported → %s", args.json)


def cmd_detect(args: argparse.Namespace) -> None:
    """Run inference on a single image."""
    from .detect import detect
    from .visualize import draw_detections

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights not found: %s", weights_path)
        sys.exit(1)

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error("Failed to read image: %s", input_path)
        sys.exit(1)

    detections = detect(image, weights=weights_path, conf=args.conf)

    if args.output:
        annotated = draw_detections(image, detections)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, annotated)
        logger.info("Saved annotated image → %s", args.output)

    if args.json:
        results = [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": list(d.bbox),
                "center": list(d.center),
            }
            for d in detections
        ]
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Detections exported → %s", args.json)

    for i, d in enumerate(detections, 1):
        logger.info("  %d. %s (%.2f) at (%d,%d) size=%dx%d",
                     i, d.class_name, d.confidence, *d.bbox)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phase3_yolo_detection",
        description="YOLO-based object detection for construction symbols.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic training dataset")
    p_gen.add_argument("--output", default="data/yolo_dataset", help="Dataset output directory")
    p_gen.add_argument("--num-images", type=int, default=200, help="Number of images")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")

    # train
    p_train = sub.add_parser("train", help="Fine-tune YOLOv8n")
    p_train.add_argument("--data", required=True, help="Path to data.yaml")
    p_train.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p_train.add_argument("--imgsz", type=int, default=640, help="Image size")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate trained model")
    p_eval.add_argument("--weights", required=True, help="Path to model weights")
    p_eval.add_argument("--data", required=True, help="Path to data.yaml")
    p_eval.add_argument("--json", help="Export metrics as JSON")

    # detect
    p_det = sub.add_parser("detect", help="Run inference on image")
    p_det.add_argument("--input", required=True, help="Input image path")
    p_det.add_argument("--output", help="Output annotated image path")
    p_det.add_argument("--weights", default="models/best.pt", help="Model weights path")
    p_det.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_det.add_argument("--json", help="Export detections as JSON")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "detect":
        cmd_detect(args)


if __name__ == "__main__":
    main()
