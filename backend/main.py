"""
TokenSqueeze FastAPI Backend

API endpoints for the TokenSqueeze image optimization system.
"""

import os
import base64
import io
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our components
import sys
# Ensure backend is in path for proper imports
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from optimizer.intent import classify_intent, Intent, STRATEGIES
from optimizer.processor import calculate_tokens, ImageProcessor
from optimizer.pipeline import TokenSqueezePipeline, PipelineResult

# Initialize app
app = FastAPI(
    title="TokenSqueeze API",
    description="Task-aware multimodal compression for vision LLMs",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (singleton)
pipeline: Optional[TokenSqueezePipeline] = None


def get_pipeline() -> TokenSqueezePipeline:
    """Get or create the pipeline instance."""
    global pipeline
    if pipeline is None:
        api_key = os.environ.get("TOKEN_COMPANY_API_KEY")
        pipeline = TokenSqueezePipeline(api_key)
    return pipeline


# Response models
class IntentResponse(BaseModel):
    intent: str
    confidence: float
    method: str
    strategy: dict


class OptimizeResponse(BaseModel):
    # Intent
    intent: str
    intent_confidence: float
    intent_method: str

    # Image optimization
    original_size: list  # [width, height]
    processed_size: list
    was_cropped: bool
    crop_size: Optional[list]
    operations: list

    # Token savings
    original_image_tokens: int
    optimized_image_tokens: int
    image_tokens_saved: int
    image_savings_percent: float

    # Text compression
    original_text_tokens: Optional[int]
    compressed_text_tokens: Optional[int]

    # Total savings
    total_original_tokens: int
    total_optimized_tokens: int
    total_tokens_saved: int
    total_savings_percent: float

    # Cost impact
    cost_saved_per_request: float
    monthly_savings_10k: float

    # Optimized payload
    detail_mode: str
    optimized_prompt: str
    base64_image: str


class CompareRequest(BaseModel):
    query: str
    image_base64: str


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "TokenSqueeze API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/classify", response_model=IntentResponse)
async def classify_query(query: str = Form(...)):
    """
    Classify a query's intent.

    Returns the detected intent, confidence, and recommended strategy.
    """
    try:
        result = classify_intent(query)
        return IntentResponse(
            intent=result.intent.value,
            confidence=result.confidence,
            method=result.method,
            strategy=result.recommended_strategy,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_image(
    query: str = Form(...),
    image: UploadFile = File(...),
    compress_text: bool = Form(True),
    text_aggressiveness: float = Form(0.5),
):
    """
    Optimize an image based on the query intent.

    This is the main endpoint for TokenSqueeze. It:
    1. Classifies the query intent
    2. Applies appropriate image optimizations
    3. Optionally compresses the text prompt
    4. Returns the optimized payload with savings metrics
    """
    try:
        # Read image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Process through pipeline
        pipe = get_pipeline()
        result = pipe.process(
            query=query,
            image=pil_image,
            compress_text=compress_text,
            text_aggressiveness=text_aggressiveness,
        )

        # Build response
        return OptimizeResponse(
            # Intent
            intent=result.intent.value,
            intent_confidence=result.intent_confidence,
            intent_method=result.intent_method,

            # Image
            original_size=list(result.image_result.original_size),
            processed_size=list(result.image_result.processed_size),
            was_cropped=result.image_result.was_cropped,
            crop_size=list(result.image_result.crop_size) if result.image_result.crop_size else None,
            operations=result.image_result.operations_applied,

            # Image tokens
            original_image_tokens=result.image_result.original_tokens,
            optimized_image_tokens=result.image_result.optimized_tokens,
            image_tokens_saved=result.image_result.tokens_saved,
            image_savings_percent=result.image_result.savings_percent,

            # Text tokens
            original_text_tokens=result.text_result.original_tokens if result.text_result else None,
            compressed_text_tokens=result.text_result.compressed_tokens if result.text_result else None,

            # Totals
            total_original_tokens=result.total_original_tokens,
            total_optimized_tokens=result.total_optimized_tokens,
            total_tokens_saved=result.total_tokens_saved,
            total_savings_percent=result.total_savings_percent,

            # Cost
            cost_saved_per_request=result.cost_saved,
            monthly_savings_10k=result.monthly_savings_10k,

            # Payload
            detail_mode=result.detail_mode,
            optimized_prompt=result.optimized_prompt,
            base64_image=result.base64_image,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_original_vs_optimized(
    query: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Compare original vs optimized token costs for an image.

    Returns both the original and optimized token counts
    for side-by-side comparison in the demo.
    """
    try:
        # Read image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Calculate original tokens (high detail, full resolution)
        width, height = pil_image.size
        original_tokens = calculate_tokens(width, height, "high")

        # Process through pipeline for optimized version
        pipe = get_pipeline()
        result = pipe.process(query=query, image=pil_image)

        # Original payload (full image, no compression)
        original_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return {
            "query": query,
            "original": {
                "size": [width, height],
                "tokens": original_tokens,
                "detail_mode": "high",
                "prompt": query,
                "base64_image": original_base64,
            },
            "optimized": {
                "size": list(result.image_result.processed_size),
                "tokens": result.total_optimized_tokens,
                "detail_mode": result.detail_mode,
                "prompt": result.optimized_prompt,
                "base64_image": result.base64_image,
                "was_cropped": result.image_result.was_cropped,
                "operations": result.image_result.operations_applied,
            },
            "savings": {
                "tokens_saved": result.total_tokens_saved,
                "percent_saved": result.total_savings_percent,
                "cost_saved": result.cost_saved,
                "monthly_savings_10k": result.monthly_savings_10k,
            },
            "intent": {
                "detected": result.intent.value,
                "confidence": result.intent_confidence,
                "method": result.intent_method,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/intents")
async def list_intents():
    """
    List all supported intents and their strategies.
    """
    return {
        intent.value: {
            "description": strategy.get("description", ""),
            "detail_mode": strategy.get("detail_mode"),
            "max_dimension": strategy.get("max_dimension"),
            "grayscale": strategy.get("grayscale"),
            "jpeg_quality": strategy.get("jpeg_quality"),
            "smart_crop": strategy.get("smart_crop", False),
        }
        for intent, strategy in STRATEGIES.items()
    }


class CallModelRequest(BaseModel):
    prompt: str
    base64_image: str
    detail_mode: str = "high"


@app.post("/call-model")
async def call_vision_model(request: CallModelRequest):
    """
    Call GPT-4o vision model with the given prompt and image.

    Used by the demo to show both original and optimized get same answer.
    """
    import requests as http_requests

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        return {"error": "OPENROUTER_API_KEY not set", "response": None}

    try:
        response = http_requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{request.base64_image}",
                                    "detail": request.detail_mode,
                                },
                            },
                        ],
                    }
                ],
            },
            timeout=60,
        )

        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            return {"response": data["choices"][0]["message"]["content"], "error": None}
        else:
            return {"error": data.get("error", {}).get("message", "Unknown error"), "response": None}

    except Exception as e:
        return {"error": str(e), "response": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
