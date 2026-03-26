"""Image preprocessing for OCR: deskew, denoise, and adaptive thresholding.

Prepares document images for Tesseract by normalizing orientation, reducing
noise, and binarizing text against background.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale. Returns as-is if already single-channel."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove noise using non-local means denoising.

    Args:
        image: Grayscale image.

    Returns:
        Denoised grayscale image.
    """
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Binarize an image using Otsu's thresholding with Gaussian blur.

    Args:
        image: Grayscale image.

    Returns:
        Binary image (white text regions on black, or vice versa depending on input).
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def detect_skew_angle(image: np.ndarray) -> float:
    """Detect the skew angle of text in a binarized image.

    Uses minAreaRect on all non-zero points to find the dominant text angle.

    Args:
        image: Grayscale or binary image.

    Returns:
        Skew angle in degrees (negative = clockwise skew).
    """
    # Invert if needed so text pixels are white
    if np.mean(image) > 127:
        work = cv2.bitwise_not(image)
    else:
        work = image.copy()

    # Threshold to ensure binary
    _, binary = cv2.threshold(work, 127, 255, cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # minAreaRect returns angles in [-90, 0). Normalize to [-45, 45].
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    return float(angle)


def deskew(image: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Rotate an image to correct skew.

    Args:
        image: Input image (grayscale or BGR).
        angle: Skew angle in degrees. If None, auto-detected.

    Returns:
        Deskewed image.
    """
    if angle is None:
        gray = to_grayscale(image) if len(image.shape) == 3 else image
        angle = detect_skew_angle(gray)

    if abs(angle) < 0.5:
        logger.info("Skew angle %.2f° is negligible, skipping rotation", angle)
        return image

    logger.info("Correcting skew angle: %.2f°", angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess(image: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline: grayscale → denoise → deskew → threshold.

    Args:
        image: Input BGR image.

    Returns:
        Preprocessed binary image ready for OCR.
    """
    gray = to_grayscale(image)
    logger.info("Converted to grayscale: %dx%d", gray.shape[1], gray.shape[0])

    denoised = denoise(gray)
    logger.info("Applied denoising")

    deskewed = deskew(denoised)

    binary = adaptive_threshold(deskewed)
    logger.info("Applied adaptive thresholding")

    return binary
