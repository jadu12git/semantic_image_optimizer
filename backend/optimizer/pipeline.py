"""
TokenSqueeze Full Pipeline

Combines:
1. Intent classification (query → intent → strategy)
2. Image processing (image + strategy → optimized image)
3. Text compression (Token Company API)
4. Token calculation (before/after savings)

This is the main entry point for the optimization system.
"""

import base64
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
from PIL import Image

from .intent import IntentClassifier, IntentResult, Intent
from .processor import ImageProcessor, ProcessingResult, calculate_tokens

# Handle both direct imports and package imports
try:
    from ..tokens import TokenCompanyClient, CompressionResult
except ImportError:
    from tokens import TokenCompanyClient, CompressionResult


@dataclass
class PipelineResult:
    """Complete result of the optimization pipeline"""

    # Intent classification
    intent: Intent
    intent_confidence: float
    intent_method: str

    # Image processing
    image_result: ProcessingResult

    # Text compression
    text_result: Optional[CompressionResult]

    # Combined savings
    total_original_tokens: int
    total_optimized_tokens: int
    total_tokens_saved: int
    total_savings_percent: float

    # Cost estimates (at $2.50/1M tokens)
    original_cost: float
    optimized_cost: float
    cost_saved: float
    monthly_savings_10k: float  # At 10K images/day

    # For API call
    detail_mode: str
    optimized_prompt: str
    base64_image: str


class TokenSqueezePipeline:
    """
    Main pipeline for TokenSqueeze optimization.

    Takes a query and image, returns optimized versions
    with token savings calculations.
    """

    def __init__(self, token_company_api_key: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            token_company_api_key: Optional API key for Token Company.
                                   If not provided, text compression is simulated.
        """
        print("Initializing TokenSqueeze Pipeline...")
        self.intent_classifier = IntentClassifier()
        self.image_processor = ImageProcessor()
        self.token_client = TokenCompanyClient(token_company_api_key)
        print("Pipeline ready.")

    def process(
        self,
        query: str,
        image: Union[Image.Image, str, Path],
        compress_text: bool = True,
        text_aggressiveness: float = 0.5,
    ) -> PipelineResult:
        """
        Process a query and image through the full pipeline.

        Args:
            query: User's query text
            image: PIL Image or path to image file
            compress_text: Whether to compress the query text
            text_aggressiveness: Compression level for text (0.0-1.0)

        Returns:
            PipelineResult with all optimization metrics
        """
        # Step 1: Classify intent
        intent_result = self.intent_classifier.classify(query)

        # Step 2: Process image based on strategy
        strategy = intent_result.recommended_strategy
        image_result = self.image_processor.process(image, strategy)

        # Step 3: Compress text (optional)
        text_result = None
        if compress_text:
            text_result = self.token_client.compress(query, text_aggressiveness)
            optimized_prompt = text_result.compressed_text
        else:
            optimized_prompt = query

        # Step 4: Calculate combined savings
        # Image tokens
        image_original = image_result.original_tokens
        image_optimized = image_result.optimized_tokens

        # Text tokens (rough estimate if not compressed)
        if text_result:
            text_original = text_result.original_tokens
            text_optimized = text_result.compressed_tokens
        else:
            # Estimate: ~1.3 tokens per word
            words = len(query.split())
            text_original = int(words * 1.3)
            text_optimized = text_original

        # Totals
        total_original = image_original + text_original
        total_optimized = image_optimized + text_optimized
        total_saved = total_original - total_optimized
        savings_percent = (total_saved / total_original * 100) if total_original > 0 else 0

        # Cost calculations (GPT-4o: $2.50/1M input tokens)
        cost_per_token = 2.50 / 1_000_000
        original_cost = total_original * cost_per_token
        optimized_cost = total_optimized * cost_per_token
        cost_saved = total_saved * cost_per_token
        monthly_savings = cost_saved * 10_000 * 30  # 10K images/day for 30 days

        return PipelineResult(
            # Intent
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            intent_method=intent_result.method,

            # Image
            image_result=image_result,

            # Text
            text_result=text_result,

            # Combined savings
            total_original_tokens=total_original,
            total_optimized_tokens=total_optimized,
            total_tokens_saved=total_saved,
            total_savings_percent=savings_percent,

            # Cost
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            cost_saved=cost_saved,
            monthly_savings_10k=monthly_savings,

            # For API call
            detail_mode=image_result.detail_mode,
            optimized_prompt=optimized_prompt,
            base64_image=image_result.base64_image,
        )

    def get_api_payload(self, result: PipelineResult, model: str = "gpt-4o") -> dict:
        """
        Generate the API payload for a vision model call.

        Args:
            result: PipelineResult from process()
            model: Model to use (default: gpt-4o)

        Returns:
            Dict ready to send to OpenAI API
        """
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": result.optimized_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{result.base64_image}",
                                "detail": result.detail_mode,
                            },
                        },
                    ],
                }
            ],
        }


# Singleton instance
_pipeline_instance: Optional[TokenSqueezePipeline] = None


def get_pipeline(api_key: Optional[str] = None) -> TokenSqueezePipeline:
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TokenSqueezePipeline(api_key)
    return _pipeline_instance


def optimize(
    query: str,
    image: Union[Image.Image, str, Path],
    compress_text: bool = True,
    text_aggressiveness: float = 0.5,
) -> PipelineResult:
    """
    Convenience function to run the full optimization pipeline.

    Args:
        query: User's query
        image: Image to optimize
        compress_text: Whether to compress text
        text_aggressiveness: Text compression level

    Returns:
        PipelineResult
    """
    pipeline = get_pipeline()
    return pipeline.process(query, image, compress_text, text_aggressiveness)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <query> <image_path>")
        print("Example: python pipeline.py 'Is there a dog?' image.jpg")
        sys.exit(1)

    query = sys.argv[1]
    image_path = sys.argv[2]

    pipeline = TokenSqueezePipeline()
    result = pipeline.process(query, image_path)

    print("\n" + "=" * 70)
    print("TOKENSQUEEZE PIPELINE RESULT")
    print("=" * 70)

    print(f"\n[INTENT]")
    print(f"  Detected: {result.intent.value}")
    print(f"  Confidence: {result.intent_confidence:.1%}")
    print(f"  Method: {result.intent_method}")

    print(f"\n[IMAGE OPTIMIZATION]")
    print(f"  Original: {result.image_result.original_size} → {result.image_result.original_tokens} tokens")
    print(f"  Optimized: {result.image_result.processed_size} → {result.image_result.optimized_tokens} tokens")
    print(f"  Savings: {result.image_result.tokens_saved} tokens ({result.image_result.savings_percent:.1f}%)")

    if result.text_result:
        print(f"\n[TEXT COMPRESSION]")
        print(f"  Original: {result.text_result.original_tokens} tokens")
        print(f"  Compressed: {result.text_result.compressed_tokens} tokens")
        print(f"  Savings: {result.text_result.tokens_saved} tokens ({result.text_result.savings_percent:.1f}%)")

    print(f"\n[TOTAL SAVINGS]")
    print(f"  Original: {result.total_original_tokens} tokens")
    print(f"  Optimized: {result.total_optimized_tokens} tokens")
    print(f"  Saved: {result.total_tokens_saved} tokens ({result.total_savings_percent:.1f}%)")

    print(f"\n[COST IMPACT]")
    print(f"  Per request: ${result.cost_saved:.6f} saved")
    print(f"  At 10K/day for 30 days: ${result.monthly_savings_10k:.2f} saved")

    print(f"\n[OUTPUT]")
    print(f"  Detail mode: {result.detail_mode}")
    print(f"  Prompt: {result.optimized_prompt[:100]}...")
    print(f"  Image base64: {len(result.base64_image)} chars")
