# TokenSqueeze: Task-Aware Multimodal Compression for Vision LLMs

## What This Is

A preprocessing layer that reduces vision API costs by intelligently optimizing images based on what the user is actually trying to do. The insight: most vision API calls send way more visual data than the task requires.

**Hackathon Project** — Token Company Track. Goal: demonstrate significant token savings while maintaining model output quality.

---

## Core Insight

GPT-4o charges for images based on resolution and detail level:
- **Low detail mode**: Fixed 85 tokens, regardless of image size
- **High detail mode**: Scales with image dimensions (170 tokens per 512x512 tile + 85 base)

A 2048x4096 image costs ~1,105 tokens. But if someone just wants to know "is there text in this image?" — that's a binary question that could be answered with 85 tokens in low detail mode. That's **92% savings** for the same answer quality.

The key is knowing *what the user wants* so we can optimize appropriately.

---

## Architecture (Finalized)

```
User Query + Image
       ↓
┌─────────────────────────────────────┐
│  CLIP-based Intent + Image Analysis │  ← Runs locally, ZERO API cost
│  - Query embedding                  │
│  - Image embedding                  │
│  - Intent classification            │
│  - Image content understanding      │
└─────────────────────────────────────┘
       ↓
Strategy Selection (lookup table)
       ↓
Image Processor (Pillow)
       ↓
Token Calculator (before/after)
       ↓
[Optional] Token Company API (compress text prompt)
       ↓
Vision API (GPT-4o) with optimized image
       ↓
Response + Savings Report
```

**Key Design Decision**: No LLM in the preprocessing pipeline. Using an LLM to classify intent would defeat the purpose (spending tokens to save tokens). CLIP runs locally with zero API cost.

---

## Why CLIP

CLIP (Contrastive Language-Image Pre-training) is perfect for this use case:

1. **Encodes text and images into the same embedding space** — we can compare query intent to canonical examples
2. **Runs locally** — no API calls, no token costs
3. **Fast inference** — ~50-100ms
4. **Understands both query AND image content** — enables smarter optimization decisions

Model: `openai/clip-vit-base-patch32` (fast, good enough for demo)

---

## Intent Classification

### Intent Categories

| Intent | Description | Optimization Strategy |
|--------|-------------|----------------------|
| `BINARY_QUESTION` | Yes/no questions about image content | Maximum compression, low detail mode (85 tokens) |
| `TEXT_EXTRACTION` | OCR, reading text from images | High res in text regions, grayscale OK |
| `SCENE_DESCRIPTION` | General "what's in this image" queries | Moderate compression, keep color |
| `OBJECT_DETECTION` | Finding/counting specific things | Can compress, need object visibility |
| `DETAILED_ANALYSIS` | Charts, graphs, technical content | Minimal compression, preserve everything |
| `COMPARISON` | Comparing multiple images | Consistent processing across images |

### How CLIP Classification Works

1. Pre-compute embeddings for canonical descriptions of each intent
2. Embed the user's query
3. Calculate cosine similarity between query embedding and intent embeddings
4. Return highest-scoring intent + confidence

Example canonical descriptions:
```python
INTENT_DESCRIPTIONS = {
    "BINARY_QUESTION": "Answer yes or no. Is there something in the image? Does it contain something?",
    "TEXT_EXTRACTION": "Read the text. Extract words. OCR. Transcribe. What does it say?",
    "SCENE_DESCRIPTION": "Describe this image. What is in this picture? Tell me about this.",
    ...
}
```

---

## Optimization Strategies

```python
STRATEGIES = {
    "BINARY_QUESTION": {
        "detail_mode": "low",      # 85 tokens flat
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
    "DETAILED_ANALYSIS": {
        "detail_mode": "high",
        "max_dimension": 2048,
        "grayscale": False,
        "jpeg_quality": 90,
    },
}
```

---

## Token Calculation (GPT-4o)

### Low Detail Mode
Always **85 tokens**. Period.

### High Detail Mode
1. Scale image to fit within 2048x2048 (preserve aspect ratio)
2. Scale so shortest side is 768px
3. Count 512x512 tiles
4. Total = `(tile_count × 170) + 85`

```python
def calculate_tokens(width, height, detail="high"):
    if detail == "low":
        return 85

    # Scale to fit 2048x2048
    if max(width, height) > 2048:
        scale = 2048 / max(width, height)
        width, height = int(width * scale), int(height * scale)

    # Scale shortest side to 768
    scale = 768 / min(width, height)
    width, height = int(width * scale), int(height * scale)

    # Count tiles
    tiles_x = math.ceil(width / 512)
    tiles_y = math.ceil(height / 512)

    return 85 + (170 * tiles_x * tiles_y)
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Intent Classification | CLIP (`openai/clip-vit-base-patch32`) |
| Image Processing | Pillow |
| Token Calculation | Custom (OpenAI formula) |
| Text Compression | Token Company API (`bear-1`) |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## Token Company Integration

Their API compresses text prompts. We compress images. Together = full multimodal compression.

**Endpoint**: `POST https://api.thetokencompany.com/v1/compress`
**Model**: `bear-1`
**Key parameter**: `aggressiveness` (0.0-1.0)

---

## Demo Flow (For Judges)

1. **Show the problem**: Large image + simple question = wasted tokens
2. **Show the solution**: Intent detection → smart compression → massive savings
3. **Show the math**: At scale, this saves $X/month
4. **Let them try**: Interactive demo with their own images

---

## File Structure

```
semantic_image_optimizer/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── intent.py        # CLIP-based intent classification ✅ DONE
│   │   ├── strategies.py    # Compression strategies per intent
│   │   └── processor.py     # Pillow image operations
│   ├── api/
│   │   └── vision.py        # GPT-4o wrapper + token counting
│   └── tokens.py            # Token calculation helpers
├── frontend/
│   └── app.py               # Streamlit app
├── tests/
│   └── sample_images/       # Test images
├── test_intent.py           # Test script for intent classifier ✅ DONE
├── requirements.txt         # ✅ DONE
└── README.md
```

---

## Success Metrics

- **Token reduction**: 40-70% average across mixed workloads
- **Accuracy preservation**: Optimized responses match unoptimized responses
- **Demo clarity**: Judges immediately understand the value prop
- **Wow factor**: The savings numbers should be impressive
