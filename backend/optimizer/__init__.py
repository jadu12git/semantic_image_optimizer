"""
TokenSqueeze Optimizer Module

Components:
- intent: Query intent classification
- processor: Image optimization
- pipeline: Full optimization pipeline
"""

from .intent import IntentClassifier, Intent, IntentResult, classify_intent
from .processor import ImageProcessor, ProcessingResult, process_image, calculate_tokens
from .cropper import smart_crop, CropResult

# Lazy load pipeline to avoid import issues
def _get_pipeline_imports():
    from .pipeline import TokenSqueezePipeline, PipelineResult, optimize, get_pipeline
    return TokenSqueezePipeline, PipelineResult, optimize, get_pipeline

__all__ = [
    # Intent
    "IntentClassifier",
    "Intent",
    "IntentResult",
    "classify_intent",
    # Processor
    "ImageProcessor",
    "ProcessingResult",
    "process_image",
    "calculate_tokens",
    # Cropper
    "smart_crop",
    "CropResult",
]
