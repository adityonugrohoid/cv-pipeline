"""Draw detected shapes on an image with bounding boxes, centers, and labels."""

import cv2
import numpy as np

from .detector import Shape

# Color palette (BGR) — one per shape type
SHAPE_COLORS: dict[str, tuple[int, int, int]] = {
    "rectangle": (0, 180, 0),    # green
    "square": (0, 220, 0),       # bright green
    "circle": (200, 0, 0),       # blue
    "triangle": (0, 0, 200),     # red
    "polygon": (180, 100, 50),   # teal
}

DEFAULT_COLOR = (128, 128, 128)


def annotate_image(image: np.ndarray, shapes: list[Shape]) -> np.ndarray:
    """Draw bounding boxes, center dots, and labels on a copy of the image.

    Args:
        image: BGR image as numpy array.
        shapes: List of detected Shape objects.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()

    for shape in shapes:
        color = SHAPE_COLORS.get(shape.shape_type, DEFAULT_COLOR)
        x, y, w, h = shape.bbox
        cx, cy = shape.center

        # Draw contour
        cv2.drawContours(annotated, [shape.contour], -1, color, 2)

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)

        # Center dot
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        # Label: shape type + key dimension
        if shape.shape_type == "circle":
            label = f"circle r={shape.properties.get('radius', '?')}"
        elif shape.shape_type in ("rectangle", "square"):
            label = f"{shape.shape_type} {w}x{h}"
        elif shape.shape_type == "triangle":
            label = f"triangle {w}x{h}"
        else:
            verts = shape.properties.get("vertices", "?")
            label = f"polygon({verts}) {w}x{h}"

        cv2.putText(
            annotated, label, (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    return annotated
