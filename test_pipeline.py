#!/usr/bin/env python3
"""
Test the full TokenSqueeze pipeline:
1. Intent classification (query → intent → strategy)
2. Image processing (image + strategy → optimized image + token savings)

Usage:
    python test_pipeline.py "your query" path/to/image.jpg
    python test_pipeline.py "Is there a dog in this?" photo.jpg
    python test_pipeline.py "Read the text" receipt.png
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from PIL import Image
from optimizer.intent import IntentClassifier
from optimizer.processor import ImageProcessor, calculate_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Test the full TokenSqueeze pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="The query to classify")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--save", "-s", help="Save processed image to this path")
    parser.add_argument("--show", action="store_true", help="Display images (requires display)")

    args = parser.parse_args()

    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Initialize components
    print("Initializing TokenSqueeze...")
    intent_classifier = IntentClassifier()
    image_processor = ImageProcessor()
    print()

    # Load original image
    original_image = Image.open(image_path)
    print("=" * 70)
    print("TOKENSQUEEZE PIPELINE")
    print("=" * 70)

    # Step 1: Intent Classification
    print(f"\n[1] INTENT CLASSIFICATION")
    print(f"    Query: \"{args.query}\"")

    intent_result = intent_classifier.classify(args.query)

    print(f"    → Intent: {intent_result.intent.value}")
    print(f"    → Confidence: {intent_result.confidence:.1%}")
    print(f"    → Method: {intent_result.method}")

    # Step 2: Strategy Selection
    print(f"\n[2] STRATEGY SELECTION")
    strategy = intent_result.recommended_strategy
    print(f"    → {strategy['description']}")
    print(f"    → Detail mode: {strategy['detail_mode']}")
    print(f"    → Max dimension: {strategy['max_dimension']}px")
    print(f"    → Grayscale: {strategy['grayscale']}")
    print(f"    → JPEG quality: {strategy['jpeg_quality']}")

    # Step 3: Image Processing
    print(f"\n[3] IMAGE PROCESSING")
    print(f"    Input: {image_path.name} ({original_image.size[0]}x{original_image.size[1]})")

    result = image_processor.process(original_image, strategy)

    print(f"    Output: {result.processed_size[0]}x{result.processed_size[1]}")
    print(f"    Operations: {', '.join(result.operations_applied)}")

    # Step 4: Token Savings
    print(f"\n[4] TOKEN SAVINGS")
    print(f"    Original:  {result.original_tokens:,} tokens")
    print(f"    Optimized: {result.optimized_tokens:,} tokens")
    print(f"    Saved:     {result.tokens_saved:,} tokens ({result.savings_percent:.1f}%)")

    # File size savings
    original_kb = result.original_bytes / 1024
    processed_kb = result.processed_bytes / 1024
    size_reduction = (1 - result.processed_bytes / result.original_bytes) * 100

    print(f"\n[5] FILE SIZE")
    print(f"    Original:  {original_kb:.1f} KB")
    print(f"    Optimized: {processed_kb:.1f} KB")
    print(f"    Reduction: {size_reduction:.1f}%")

    # Cost savings estimate
    # GPT-4o pricing: ~$2.50 per 1M input tokens
    cost_per_token = 2.50 / 1_000_000
    original_cost = result.original_tokens * cost_per_token
    optimized_cost = result.optimized_tokens * cost_per_token
    saved_cost = result.tokens_saved * cost_per_token

    print(f"\n[6] COST ESTIMATE (at $2.50/1M tokens)")
    print(f"    Original:  ${original_cost:.6f} per image")
    print(f"    Optimized: ${optimized_cost:.6f} per image")
    print(f"    Saved:     ${saved_cost:.6f} per image")
    print(f"    At 10K images/day: ${saved_cost * 10000 * 30:.2f}/month saved")

    print("\n" + "=" * 70)

    # Save processed image if requested
    if args.save:
        save_path = Path(args.save)
        result.image.save(save_path, "JPEG", quality=strategy['jpeg_quality'])
        print(f"Saved processed image to: {save_path}")

    # Show images if requested
    if args.show:
        try:
            original_image.show(title="Original")
            result.image.show(title="Optimized")
        except Exception as e:
            print(f"Could not display images: {e}")


if __name__ == "__main__":
    main()
