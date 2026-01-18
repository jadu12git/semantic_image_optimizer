# TokenSqueeze - Context for New Chat Sessions

## HACKATHON CONTEXT - IMPORTANT!
- **Event**: Currently at a hackathon with ~14 hours remaining (as of last session)
- **Sponsored Track**: Token Company track
- **Goal**: Win/develop something intuitive for judges
- **Demo Strategy**: Side-by-side comparison showing same answer with fewer tokens
- **Token Company API**: `POST https://api.thetokencompany.com/v1/compress` with `bear-1` model

## Project Overview
**TokenSqueeze** is a preprocessing layer that reduces vision API costs by intelligently compressing images based on user intent.

**Core Insight**: GPT-4o charges based on image resolution. A 2048x4096 image costs ~1,105 tokens, but a simple yes/no question can be answered with 85 tokens in low detail mode. That's 92% savings for the same answer.

**Why This Matters for Judges**:
- Real cost savings companies can use TODAY
- Uses Token Company's API for text compression (sponsor integration)
- Clear demo: same question, same answer, 88% fewer tokens

## Current Status: BACKEND COMPLETE ✅

### What's Built

1. **Intent Classifier** (`backend/optimizer/intent.py`)
   - Hybrid approach: keyword regex (95% confidence) + DeBERTa zero-shot fallback
   - 6 intents: BINARY_QUESTION, TEXT_EXTRACTION, SCENE_DESCRIPTION, OBJECT_DETECTION, DETAILED_ANALYSIS, COMPARISON

2. **Image Processor** (`backend/optimizer/processor.py`)
   - Smart crop → Resize → Grayscale → JPEG compression
   - Token calculation following GPT-4o's formula

3. **Smart Cropper** (`backend/optimizer/cropper.py`)
   - OpenCV-based document/receipt detection
   - Multiple strategies: Canny edge, adaptive threshold, white region
   - Perspective transform for clean crops

4. **Token Company Integration** (`backend/tokens.py`)
   - Text prompt compression via their API
   - API key from `.env` file

5. **Full Pipeline** (`backend/optimizer/pipeline.py`)
   - Combines all components
   - Returns comprehensive savings metrics

6. **FastAPI Backend** (`backend/main.py`)
   - Endpoints:
     - `GET /` - Health check
     - `GET /health` - Health status
     - `GET /intents` - List all intents and strategies
     - `POST /classify` - Classify query intent
     - `POST /optimize` - Full optimization pipeline
     - `POST /compare` - Side-by-side comparison (FOR DEMO)

### Test Results
- Receipt image: **88% token savings** (765 → 91 tokens)
- Smart cropping: 4032x3024 → 3400x1984 (focused on receipt)
- Monthly savings at 10K/day: **$512.25**

### To Start the Server
```bash
cd /Users/saha/Desktop/semantic_image_optimizer/backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## What's Next: FRONTEND

The user will build a Next.js frontend. Key requirements mentioned:
- **Side-by-side demo**: Same prompt and image sent to model twice
  - One passes through TokenSqueeze optimizer
  - One goes in raw
- Show both get the same answer but with different token costs
- The `/compare` endpoint is specifically designed for this

## File Structure
```
semantic_image_optimizer/
├── backend/
│   ├── main.py              # FastAPI app ✅
│   ├── tokens.py            # Token Company client ✅
│   └── optimizer/
│       ├── __init__.py
│       ├── intent.py        # Intent classification ✅
│       ├── processor.py     # Image processing ✅
│       ├── cropper.py       # Smart document cropping ✅
│       └── pipeline.py      # Full pipeline ✅
├── tests/
│   └── sample_images/
│       └── reciept.jpeg     # Test image
├── .env                     # API keys (TOKEN_COMPANY_API_KEY, OPENROUTER_API_KEY)
├── requirements.txt
├── test_api.py              # API endpoint tests
├── test_pipeline.py
└── test_with_model.py       # End-to-end with real GPT-4o
```

## API Endpoints Detail

### POST /optimize
```python
# Request
FormData:
  - query: str (user's question)
  - image: File (the image)
  - compress_text: bool (default: true)
  - text_aggressiveness: float (0.0-1.0, default: 0.5)

# Response
{
  "intent": "BINARY_QUESTION",
  "intent_confidence": 0.95,
  "original_size": [4032, 3024],
  "processed_size": [512, 298],
  "was_cropped": true,
  "operations": ["smart crop...", "resize...", "grayscale", "JPEG quality 60"],
  "original_image_tokens": 765,
  "optimized_image_tokens": 85,
  "total_savings_percent": 88.2,
  "detail_mode": "low",
  "optimized_prompt": "...",
  "base64_image": "..."
}
```

### POST /compare (FOR DEMO)
```python
# Returns both original and optimized payloads for side-by-side comparison
{
  "original": {
    "size": [4032, 3024],
    "tokens": 765,
    "detail_mode": "high",
    "base64_image": "..."
  },
  "optimized": {
    "size": [512, 298],
    "tokens": 91,
    "detail_mode": "low",
    "was_cropped": true,
    "base64_image": "..."
  },
  "savings": {
    "tokens_saved": 683,
    "percent_saved": 88.2,
    "monthly_savings_10k": 512.25
  },
  "intent": {
    "detected": "BINARY_QUESTION",
    "confidence": 0.95
  }
}
```

## Key Design Decisions
1. **No LLM in preprocessing** - Using LLM to classify would defeat the purpose
2. **Hybrid intent classification** - Keyword matching first (fast), DeBERTa fallback (accurate)
3. **Smart cropping enabled** for BINARY_QUESTION and TEXT_EXTRACTION intents
4. **Low detail mode (85 tokens)** for binary questions - massive savings

## Dependencies
```
fastapi
uvicorn
python-multipart
python-dotenv
Pillow
opencv-python
numpy
torch
transformers
requests
```

## Environment Variables (.env)
```
TOKEN_COMPANY_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here  # For calling GPT-4o
```

## Demo Flow for Judges
1. **Show the problem**: Large image + simple question = wasted tokens
2. **Show the solution**: Intent detection → smart compression → massive savings
3. **Show the math**: At scale, this saves $X/month
4. **Let them try**: Interactive demo with their own images

## Verified End-to-End Test
Already tested with real GPT-4o via OpenRouter:
- Query: "Is the total more than $25?"
- Image: Receipt (reciept.jpeg)
- Original: 816 tokens → Answer: "Yes, the total is $26.75"
- Optimized: 136 tokens → Answer: "Yes, the total is $26.75"
- **SAME ANSWER, 83% fewer tokens!**

## User Preferences
- User will build the Next.js frontend themselves
- Backend should expose clean REST endpoints (done)
- Focus on impressive demo for judges
