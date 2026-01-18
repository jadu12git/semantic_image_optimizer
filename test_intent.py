#!/usr/bin/env python3
"""
Test script for the intent classifier.

Usage:
    python test_intent.py "your query here"
    python test_intent.py --interactive

Examples:
    python test_intent.py "Is there a dog in this image?"
    python test_intent.py "Read the text on this receipt"
    python test_intent.py "How many people are in this photo?"
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from optimizer.intent import IntentClassifier, Intent


def format_scores(scores: dict) -> str:
    """Format scores in a readable way."""
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    lines = []
    for intent, score in sorted_scores:
        bar_length = int(score * 40)  # Scale to 40 chars max
        bar = "█" * bar_length + "░" * (40 - bar_length)
        lines.append(f"    {intent:20} {bar} {score:.1%}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Test the intent classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_intent.py "Is there a dog in this image?"
    python test_intent.py "Read the text on this receipt"
    python test_intent.py "How many people are in this photo?"
    python test_intent.py --interactive
        """
    )
    parser.add_argument("query", nargs="?", default="test query",
                        help="The query to classify")
    parser.add_argument("--interactive", "-I", action="store_true",
                        help="Interactive mode - keep asking for queries")

    args = parser.parse_args()

    # Initialize classifier (this loads the model)
    print("Initializing DeBERTa intent classifier...")
    classifier = IntentClassifier()
    print()

    def classify_and_print(query: str):
        """Classify a query and print results."""
        print("─" * 70)
        print(f"Query: \"{query}\"")
        print("─" * 70)

        result = classifier.classify(query)

        # Calculate margin between top 2
        sorted_scores = sorted(result.all_scores.items(), key=lambda x: -x[1])
        margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        print(f"\n✓ Detected Intent: {result.intent.value}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Method: {result.method}")
        print(f"\n→ Strategy: {result.recommended_strategy['description']}")
        print(f"  - Detail mode: {result.recommended_strategy['detail_mode']}")
        print(f"  - Max dimension: {result.recommended_strategy['max_dimension']}px")
        print(f"  - Grayscale: {result.recommended_strategy['grayscale']}")
        print(f"  - JPEG quality: {result.recommended_strategy['jpeg_quality']}")
        print()

    # Run classification if not just interactive
    if not args.interactive or args.query != "test query":
        classify_and_print(args.query)

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 70)
        print("Interactive mode. Type queries to classify (Ctrl+C to exit)")
        print("=" * 70 + "\n")

        while True:
            try:
                query = input("Query> ").strip()
                if query:
                    classify_and_print(query)
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()
