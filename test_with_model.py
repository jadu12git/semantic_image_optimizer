#!/usr/bin/env python3
"""
End-to-end test: Compare model responses with original vs optimized images.

This script:
1. Sends the ORIGINAL image to GPT-4o via OpenRouter
2. Sends the OPTIMIZED image to GPT-4o via OpenRouter
3. Compares the responses to verify optimization doesn't hurt quality
"""

import os
import sys
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from backend.optimizer.pipeline import TokenSqueezePipeline
from backend.optimizer.processor import calculate_tokens


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_model(
    prompt: str,
    image_base64: str,
    detail: str = "high",
    model: str = "openai/gpt-4o"
) -> dict:
    """
    Call a vision model via OpenRouter.

    Args:
        prompt: Text prompt
        image_base64: Base64 encoded image
        detail: "low" or "high"
        model: Model to use

    Returns:
        API response dict
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": detail,
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 500,
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()


def main():
    print("=" * 70)
    print("END-TO-END TEST: Original vs Optimized Image")
    print("=" * 70)

    # Paths
    original_path = "tests/sample_images/reciept.jpeg"
    optimized_path = "tests/sample_images/reciept_optimized.jpeg"

    # Query
    query = "Is the total on this receipt more than 25 dollars? Answer yes or no, then briefly explain."

    # Check files exist
    if not Path(original_path).exists():
        print(f"Error: {original_path} not found")
        return
    if not Path(optimized_path).exists():
        print("Optimized image not found. Running pipeline first...")
        pipeline = TokenSqueezePipeline()
        result = pipeline.process(query, original_path)
        result.image_result.image.save(optimized_path, "JPEG", quality=60)

    # Get image info
    orig_img = Image.open(original_path)
    opt_img = Image.open(optimized_path)

    orig_size = Path(original_path).stat().st_size
    opt_size = Path(optimized_path).stat().st_size

    orig_tokens = calculate_tokens(orig_img.size[0], orig_img.size[1], "high")
    opt_tokens = calculate_tokens(opt_img.size[0], opt_img.size[1], "low")

    print(f"\n[IMAGE COMPARISON]")
    print(f"  Original:  {orig_img.size[0]}x{orig_img.size[1]}, {orig_size/1024:.1f}KB, ~{orig_tokens} tokens (high detail)")
    print(f"  Optimized: {opt_img.size[0]}x{opt_img.size[1]}, {opt_size/1024:.1f}KB, ~{opt_tokens} tokens (low detail)")
    print(f"  Token savings: {orig_tokens - opt_tokens} tokens ({(1 - opt_tokens/orig_tokens)*100:.1f}%)")

    print(f"\n[QUERY]")
    print(f"  {query}")

    # Test with original image
    print(f"\n[CALLING MODEL WITH ORIGINAL IMAGE...]")
    try:
        orig_b64 = image_to_base64(original_path)
        orig_response = call_vision_model(query, orig_b64, detail="high")
        orig_answer = orig_response["choices"][0]["message"]["content"]
        orig_usage = orig_response.get("usage", {})
        print(f"  Response: {orig_answer}")
        print(f"  Tokens used: {orig_usage.get('total_tokens', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")
        orig_answer = None

    # Test with optimized image
    print(f"\n[CALLING MODEL WITH OPTIMIZED IMAGE...]")
    try:
        opt_b64 = image_to_base64(optimized_path)
        opt_response = call_vision_model(query, opt_b64, detail="low")
        opt_answer = opt_response["choices"][0]["message"]["content"]
        opt_usage = opt_response.get("usage", {})
        print(f"  Response: {opt_answer}")
        print(f"  Tokens used: {opt_usage.get('total_tokens', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")
        opt_answer = None

    # Compare
    print(f"\n[COMPARISON]")
    print("=" * 70)
    if orig_answer and opt_answer:
        # Check if both give the same yes/no answer
        orig_yes = "yes" in orig_answer.lower()[:50]
        opt_yes = "yes" in opt_answer.lower()[:50]

        if orig_yes == opt_yes:
            print("✓ BOTH MODELS GAVE THE SAME ANSWER!")
            print(f"  The receipt total IS {'more' if orig_yes else 'less'} than $25")
        else:
            print("✗ ANSWERS DIFFER - optimization may have lost information")

        print(f"\n  Original answer: {orig_answer[:200]}...")
        print(f"\n  Optimized answer: {opt_answer[:200]}...")
    print("=" * 70)


if __name__ == "__main__":
    main()
