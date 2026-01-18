"""
Image Processor for TokenSqueeze

Applies optimization operations to images based on the strategy
determined by the intent classifier.

Operations:
- Smart crop (document/receipt detection)
- Resize (scale to max dimension)
- Grayscale conversion
- JPEG compression
- Detail mode selection (for API call)
"""

import io
import math
import base64
from PIL import Image
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .cropper import smart_crop, CropResult


@dataclass
class ProcessingResult:
    """Result of image processing"""
    # The processed image
    image: Image.Image

    # Image as base64 for API calls
    base64_image: str

    # Dimensions
    original_size: Tuple[int, int]  # (width, height)
    processed_size: Tuple[int, int]

    # File sizes in bytes
    original_bytes: int
    processed_bytes: int

    # Token calculations
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    savings_percent: float

    # Detail mode for API
    detail_mode: str  # "low" or "high"

    # What operations were applied
    operations_applied: list

    # Smart crop info
    was_cropped: bool = False
    crop_size: Optional[Tuple[int, int]] = None


def calculate_tokens(width: int, height: int, detail: str = "high") -> int:
    """
    Calculate GPT-4o vision tokens for an image.

    Low detail: Always 85 tokens
    High detail: Based on tiles

    Args:
        width: Image width in pixels
        height: Image height in pixels
        detail: "low" or "high"

    Returns:
        Number of tokens
    """
    if detail == "low":
        return 85

    # High detail calculation
    # Step 1: Scale to fit within 2048x2048
    if max(width, height) > 2048:
        scale = 2048 / max(width, height)
        width = int(width * scale)
        height = int(height * scale)

    # Step 2: Scale so shortest side is 768px
    if min(width, height) > 768:
        scale = 768 / min(width, height)
        width = int(width * scale)
        height = int(height * scale)

    # Step 3: Count 512x512 tiles
    tiles_x = math.ceil(width / 512)
    tiles_y = math.ceil(height / 512)
    total_tiles = tiles_x * tiles_y

    # Step 4: Calculate tokens (170 per tile + 85 base)
    return 85 + (170 * total_tiles)


def get_image_bytes(image: Image.Image, format: str = "JPEG", quality: int = 85) -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()

    # Ensure RGB for JPEG
    if format == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    image.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string."""
    image_bytes = get_image_bytes(image, format, quality)
    return base64.b64encode(image_bytes).decode("utf-8")


class ImageProcessor:
    """
    Processes images based on optimization strategy.

    Takes a strategy dict from the intent classifier and applies
    the appropriate operations to minimize tokens while preserving
    information needed for the task.
    """

    def __init__(self):
        """Initialize the processor."""
        pass

    def process(
        self,
        image: Union[Image.Image, str, Path],
        strategy: Dict,
    ) -> ProcessingResult:
        """
        Process an image according to the given strategy.

        Args:
            image: PIL Image, file path, or Path object
            strategy: Dict with keys:
                - detail_mode: "low" or "high"
                - max_dimension: int (max width/height)
                - grayscale: bool
                - jpeg_quality: int (1-100)
                - smart_crop: bool (optional, default False)

        Returns:
            ProcessingResult with processed image and metrics
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Ensure we have a copy to work with
        image = image.copy()

        # Convert to RGB if needed (for consistent processing)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # Track original stats
        original_size = image.size
        original_bytes = len(get_image_bytes(image, "JPEG", 95))
        original_tokens = calculate_tokens(
            original_size[0], original_size[1], "high"
        )

        # Track operations
        operations = []

        # Track cropping info
        was_cropped = False
        crop_size = None

        # Extract strategy parameters
        max_dim = strategy.get("max_dimension", 2048)
        grayscale = strategy.get("grayscale", False)
        jpeg_quality = strategy.get("jpeg_quality", 85)
        detail_mode = strategy.get("detail_mode", "high")
        do_smart_crop = strategy.get("smart_crop", False)

        # Apply operations
        processed = image

        # 0. Smart crop first (before other operations)
        if do_smart_crop:
            crop_result = smart_crop(processed, fallback_to_center=False)
            if crop_result.crop_detected:
                processed = crop_result.cropped_image
                was_cropped = True
                crop_size = crop_result.cropped_size
                operations.append(f"smart crop {crop_result.original_size[0]}x{crop_result.original_size[1]} → {crop_size[0]}x{crop_size[1]}")

        # 1. Resize if needed
        if max(processed.size) > max_dim:
            processed = self._resize(processed, max_dim)
            operations.append(f"resize to max {max_dim}px")

        # 2. Grayscale if requested
        if grayscale:
            processed = self._grayscale(processed)
            operations.append("convert to grayscale")

        # 3. JPEG compression (applied when converting to bytes)
        operations.append(f"JPEG quality {jpeg_quality}")

        # Calculate final stats
        processed_size = processed.size
        processed_bytes = len(get_image_bytes(processed, "JPEG", jpeg_quality))
        optimized_tokens = calculate_tokens(
            processed_size[0], processed_size[1], detail_mode
        )

        tokens_saved = original_tokens - optimized_tokens
        savings_percent = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0

        # Generate base64 for API
        base64_image = image_to_base64(processed, "JPEG", jpeg_quality)

        return ProcessingResult(
            image=processed,
            base64_image=base64_image,
            original_size=original_size,
            processed_size=processed_size,
            original_bytes=original_bytes,
            processed_bytes=processed_bytes,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=tokens_saved,
            savings_percent=savings_percent,
            detail_mode=detail_mode,
            operations_applied=operations,
            was_cropped=was_cropped,
            crop_size=crop_size,
        )

    def _resize(self, image: Image.Image, max_dimension: int) -> Image.Image:
        """
        Resize image to fit within max_dimension while preserving aspect ratio.

        Uses LANCZOS resampling for best quality.
        """
        width, height = image.size

        if max(width, height) <= max_dimension:
            return image

        # Calculate new dimensions
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        return image.resize((new_width, new_height), Image.LANCZOS)

    def _grayscale(self, image: Image.Image) -> Image.Image:
        """Convert image to grayscale."""
        # Convert to grayscale then back to RGB (for consistent format)
        gray = image.convert("L")
        return gray.convert("RGB")


def process_image(
    image: Union[Image.Image, str, Path],
    strategy: Dict
) -> ProcessingResult:
    """
    Convenience function to process an image.

    Args:
        image: PIL Image or path to image
        strategy: Optimization strategy dict

    Returns:
        ProcessingResult
    """
    processor = ImageProcessor()
    return processor.process(image, strategy)


if __name__ == "__main__":
    import sys

    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python processor.py <image_path>")
        print("\nExample strategies will be applied to show token savings.")
        sys.exit(1)

    image_path = sys.argv[1]

    # Test with different strategies
    strategies = {
        "BINARY_QUESTION": {
            "detail_mode": "low",
            "max_dimension": 512,
            "grayscale": True,
            "jpeg_quality": 60,
        },
        "TEXT_EXTRACTION": {
            "detail_mode": "high",
            "max_dimension": 2048,
            "grayscale": True,
            "jpeg_quality": 85,
        },
        "SCENE_DESCRIPTION": {
            "detail_mode": "high",
            "max_dimension": 1024,
            "grayscale": False,
            "jpeg_quality": 75,
        },
    }

    print(f"\nProcessing: {image_path}")
    print("=" * 60)

    for name, strategy in strategies.items():
        result = process_image(image_path, strategy)

        print(f"\n{name}:")
        print(f"  Original:  {result.original_size[0]}x{result.original_size[1]} → {result.original_tokens} tokens")
        print(f"  Optimized: {result.processed_size[0]}x{result.processed_size[1]} → {result.optimized_tokens} tokens")
        print(f"  Savings:   {result.tokens_saved} tokens ({result.savings_percent:.1f}%)")
        print(f"  Operations: {', '.join(result.operations_applied)}")
        print(f"  File size: {result.original_bytes:,} → {result.processed_bytes:,} bytes")
