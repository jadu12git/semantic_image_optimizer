"""
Smart Cropping for TokenSqueeze

Uses OpenCV to detect and crop documents/receipts from images.
This removes unnecessary background and focuses on the relevant content.

Approach:
1. Edge detection (Canny)
2. Find contours
3. Detect document boundary (4-point polygon)
4. Perspective transform to get a clean crop

Based on: https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CropResult:
    """Result of smart cropping"""
    cropped_image: Image.Image
    original_size: Tuple[int, int]
    cropped_size: Tuple[int, int]
    crop_detected: bool
    contour_points: Optional[np.ndarray]  # The 4 corners if detected


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order.

    Args:
        pts: Array of 4 points

    Returns:
        Ordered points array
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to get a top-down view of the document.

    Args:
        image: Input image (numpy array)
        pts: Four corner points

    Returns:
        Warped/transformed image
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Compute height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination points for "birds eye view"
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and apply
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def detect_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the document/receipt contour in an image.

    Uses multiple strategies:
    1. Standard Canny edge detection
    2. Adaptive thresholding for receipts on textured backgrounds
    3. Color-based detection (white paper detection)

    Args:
        image: Input image (BGR numpy array)

    Returns:
        4-point contour if found, None otherwise
    """
    # Resize for faster processing
    orig_height = image.shape[0]
    ratio = orig_height / 500.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Try multiple detection strategies
    contour = None

    # Strategy 1: Standard Canny edge detection
    contour = _detect_with_canny(resized)

    # Strategy 2: If Canny fails, try adaptive threshold (good for receipts)
    if contour is None:
        contour = _detect_with_adaptive_threshold(resized)

    # Strategy 3: If still nothing, try detecting white regions (paper)
    if contour is None:
        contour = _detect_white_region(resized)

    if contour is not None:
        # Scale contour back to original size
        contour = contour.reshape(4, 2) * ratio

    return contour


def _detect_with_canny(resized: np.ndarray) -> Optional[np.ndarray]:
    """Standard Canny edge detection approach with multiple thresholds."""
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple Canny thresholds (lower catches more edges)
    for low_thresh, high_thresh in [(30, 100), (50, 150), (75, 200)]:
        edged = cv2.Canny(blurred, low_thresh, high_thresh)

        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        result = _find_quad_contour(contours, resized.shape, min_area_ratio=0.2)
        if result is not None:
            return result

    return None


def _detect_with_adaptive_threshold(resized: np.ndarray) -> Optional[np.ndarray]:
    """Adaptive threshold - good for receipts with text."""
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Invert so paper is white
    thresh = cv2.bitwise_not(thresh)

    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    return _find_quad_contour(contours, resized.shape)


def _detect_white_region(resized: np.ndarray) -> Optional[np.ndarray]:
    """Detect white/light colored paper regions against darker backgrounds."""
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply strong blur to reduce texture noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Use Otsu's threshold to separate paper from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up with morphological operations
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    return _find_quad_contour(contours, resized.shape, min_area_ratio=0.15)


def _find_quad_contour(
    contours: List,
    image_shape: Tuple,
    min_area_ratio: float = 0.1,
    max_area_ratio: float = 0.9
) -> Optional[np.ndarray]:
    """Find a 4-point contour (quadrilateral) from a list of contours.

    Filters out contours that touch edges (likely background) or are too large.
    """
    image_height, image_width = image_shape[:2]
    image_area = image_height * image_width
    edge_margin = 5  # pixels from edge

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)

            # Check area bounds
            if area < image_area * min_area_ratio or area > image_area * max_area_ratio:
                continue

            # Check if contour touches image edges (likely not a document)
            pts = approx.reshape(4, 2)
            touches_edge = False
            for pt in pts:
                if (pt[0] <= edge_margin or pt[0] >= image_width - edge_margin or
                    pt[1] <= edge_margin or pt[1] >= image_height - edge_margin):
                    touches_edge = True
                    break

            if not touches_edge:
                return approx

    # Second pass: allow edge-touching but prefer smaller, more centered quads
    valid_quads = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if min_area_ratio * image_area < area < max_area_ratio * image_area:
                valid_quads.append((area, approx))

    # Return smallest valid quad (more likely to be the document, not background)
    if valid_quads:
        valid_quads.sort(key=lambda x: x[0])
        return valid_quads[0][1]

    return None


def smart_crop(image: Image.Image, fallback_to_center: bool = True) -> CropResult:
    """
    Smart crop an image to focus on the document/receipt.

    Args:
        image: PIL Image to crop
        fallback_to_center: If no document detected, do a center crop

    Returns:
        CropResult with cropped image and metadata
    """
    original_size = image.size

    # Convert PIL to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Try to detect document contour
    contour = detect_document_contour(cv_image)

    if contour is not None:
        # Document detected - apply perspective transform
        warped = four_point_transform(cv_image, contour)

        # Convert back to PIL
        cropped = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

        return CropResult(
            cropped_image=cropped,
            original_size=original_size,
            cropped_size=cropped.size,
            crop_detected=True,
            contour_points=contour,
        )

    elif fallback_to_center:
        # No document detected - do a center crop (remove 10% from each edge)
        width, height = original_size
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)

        cropped = image.crop((
            margin_x,
            margin_y,
            width - margin_x,
            height - margin_y
        ))

        return CropResult(
            cropped_image=cropped,
            original_size=original_size,
            cropped_size=cropped.size,
            crop_detected=False,
            contour_points=None,
        )

    else:
        # Return original
        return CropResult(
            cropped_image=image.copy(),
            original_size=original_size,
            cropped_size=original_size,
            crop_detected=False,
            contour_points=None,
        )


def visualize_detection(image: Image.Image, contour: Optional[np.ndarray]) -> Image.Image:
    """
    Draw the detected document contour on the image for visualization.

    Args:
        image: Original PIL Image
        contour: 4-point contour (or None)

    Returns:
        Image with contour drawn
    """
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if contour is not None:
        # Draw the contour
        pts = contour.astype(np.int32).reshape((-1, 1, 2))
        cv2.drawContours(cv_image, [pts], -1, (0, 255, 0), 3)

        # Draw corner points
        for point in contour:
            cv2.circle(cv_image, tuple(point.astype(int)), 10, (0, 0, 255), -1)

    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cropper.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path)

    print(f"Original size: {image.size}")

    result = smart_crop(image)

    print(f"Crop detected: {result.crop_detected}")
    print(f"Cropped size: {result.cropped_size}")

    if result.crop_detected:
        print(f"Contour points: {result.contour_points}")

    # Save results
    output_path = image_path.replace(".", "_cropped.")
    result.cropped_image.save(output_path)
    print(f"Saved cropped image to: {output_path}")

    # Save visualization
    viz = visualize_detection(image, result.contour_points)
    viz_path = image_path.replace(".", "_detected.")
    viz.save(viz_path)
    print(f"Saved detection visualization to: {viz_path}")
