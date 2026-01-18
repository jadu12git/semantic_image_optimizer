"""
TokenSqueeze Full Pipeline (Lite Version for Serverless)

Combines:
1. Intent classification (keyword-only, no PyTorch)
2. Image processing (image + strategy → optimized image)
3. Text compression (Token Company API)
4. Token calculation (before/after savings)
"""

import base64
import io
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
from PIL import Image

from .intent_lite import IntentClassifier, IntentResult, Intent
from .processor import ImageProcessor, ProcessingResult, calculate_tokens

# Handle both direct imports and package imports
try:
    from ..tokens import TokenCompanyClient, CompressionResult
except ImportError:
    try:
        from tokens import TokenCompanyClient, CompressionResult
    except ImportError:
        # Fallback if tokens module not available
        TokenCompanyClient = None
        CompressionResult = None


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
    text_result: Optional[any]

    # Combined savings
    total_original_tokens: int
    total_optimized_tokens: int
    total_tokens_saved: int
    total_savings_percent: float

    # Cost estimates (at $2.50/1M tokens)
    cost_saved: float
    monthly_savings_10k: float

    # Final outputs
    detail_mode: str
    optimized_prompt: str
    base64_image: str


class TokenSqueezePipeline:
    """
    Main pipeline for TokenSqueeze optimization (Lite version).
    """

    def __init__(self, token_company_api_key: Optional[str] = None):
        self.intent_classifier = IntentClassifier()
        self.image_processor = ImageProcessor()

        if token_company_api_key and TokenCompanyClient:
            self.token_client = TokenCompanyClient(api_key=token_company_api_key)
        else:
            self.token_client = None

    def process(
        self,
        query: str,
        image: Union[str, Path, Image.Image],
        compress_text: bool = True,
        text_aggressiveness: float = 0.5,
    ) -> PipelineResult:
        """
        Process an image and query through the full optimization pipeline.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        else:
            pil_image = image

        # Step 1: Classify intent
        intent_result = self.intent_classifier.classify(query)

        # Step 2: Process image based on intent strategy
        image_result = self.image_processor.process(
            pil_image, intent_result.recommended_strategy
        )

        # Step 3: Optionally compress text
        text_result = None
        optimized_prompt = query

        if compress_text and self.token_client:
            try:
                text_result = self.token_client.compress(
                    query, aggressiveness=text_aggressiveness
                )
                optimized_prompt = text_result.compressed_text
            except Exception as e:
                print(f"Text compression failed: {e}")
                optimized_prompt = query

        # Step 4: Calculate total savings
        original_text_tokens = len(query.split()) * 1.3  # rough estimate
        compressed_text_tokens = (
            text_result.compressed_tokens if text_result else original_text_tokens
        )

        total_original = image_result.original_tokens + int(original_text_tokens)
        total_optimized = image_result.optimized_tokens + int(compressed_text_tokens)
        total_saved = total_original - total_optimized
        savings_percent = (total_saved / total_original * 100) if total_original > 0 else 0

        # Cost calculation ($2.50 per 1M tokens)
        cost_per_token = 2.50 / 1_000_000
        cost_saved = total_saved * cost_per_token
        monthly_10k = cost_saved * 10_000 * 30

        # Convert processed image to base64
        img_buffer = io.BytesIO()
        image_result.processed_image.save(img_buffer, format="JPEG", quality=85)
        base64_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        return PipelineResult(
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            intent_method=intent_result.method,
            image_result=image_result,
            text_result=text_result,
            total_original_tokens=total_original,
            total_optimized_tokens=total_optimized,
            total_tokens_saved=total_saved,
            total_savings_percent=savings_percent,
            cost_saved=cost_saved,
            monthly_savings_10k=monthly_10k,
            detail_mode=intent_result.recommended_strategy.get("detail_mode", "high"),
            optimized_prompt=optimized_prompt,
            base64_image=base64_image,
        )


# Singleton
_pipeline_instance: Optional[TokenSqueezePipeline] = None


def get_pipeline(api_key: Optional[str] = None) -> TokenSqueezePipeline:
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TokenSqueezePipeline(api_key)
    return _pipeline_instance


def optimize(
    query: str,
    image: Union[str, Path, Image.Image],
    api_key: Optional[str] = None,
    compress_text: bool = True,
) -> PipelineResult:
    """Convenience function for quick optimization."""
    pipeline = get_pipeline(api_key)
    return pipeline.process(query, image, compress_text=compress_text)
