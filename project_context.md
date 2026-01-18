# TokenSqueeze: Task-Aware Multimodal Compression for Vision LLMs

## What This Is

A preprocessing layer that reduces vision API costs by intelligently optimizing images based on what the user is actually trying to do. The insight: most vision API calls send way more visual data than the task requires.

---

## Core Insight

GPT-4o charges for images based on resolution and detail level:
- **Low detail mode**: Fixed 85 tokens, regardless of image size
- **High detail mode**: Scales with image dimensions (170 tokens per 512x512 tile + 85 base)

A 2048x4096 image costs ~1,105 tokens. But if someone just wants to know "is there text in this image?" — that's a binary question that could be answered with 85 tokens in low detail mode. That's 92% savings for the same answer quality.

The key is knowing *what the user wants* so we can optimize appropriately.

---

## The Pipeline

```
User Query + Image
       ↓
Intent Classification (CRITICAL - this determines everything)
       ↓
Strategy Selection (based on intent)
       ↓
Image Optimization (resize, crop, grayscale, quality, detail mode)
       ↓
Vision API Call
       ↓
Response + Savings Report
```

---

## Intent Classification (The Hard Part)

This is the most crucial component. If we misclassify intent, we either:
- Over-compress and lose information the model needed
- Under-compress and waste tokens

### What the classifier needs to understand:

**Text Extraction Tasks**
- User wants to read/extract/transcribe text from the image
- Needs: High resolution in text regions, but color often irrelevant
- Signals: "extract", "read", "OCR", "transcribe", "what does it say", "receipt", "document"

**Scene Description Tasks**
- User wants a general understanding of what's in the image
- Needs: Moderate resolution, color matters, full image context
- Signals: "describe", "what's in this", "tell me about", "explain this image"

**Object Detection / Localization**
- User wants to find or count specific things
- Needs: Enough resolution to identify objects, but not fine detail
- Signals: "find", "locate", "is there a", "count", "how many", "detect"

**Binary Questions**
- User wants a yes/no or simple categorical answer
- Needs: Minimal detail — the model just needs to recognize presence/absence
- Signals: "is this a", "does it have", "yes or no", "true or false", "is there"

**Detailed Analysis**
- User wants deep examination (charts, diagrams, technical content)
- Needs: Maximum preservation — don't compress much
- Signals: "analyze", "chart", "graph", "diagram", "detailed", "explain the data"

**Comparison Tasks**
- User wants to compare multiple images or regions
- Needs: Consistent processing across images
- Signals: "difference", "compare", "which one", "same or different"

### The classifier should output:
1. **Intent category** — which type of task
2. **Confidence score** — how sure are we
3. **Key entities** — what specifically is the user looking for (if applicable)

---

## Optimization Strategies

Each intent maps to a strategy. The strategy defines:

| Parameter | What it controls |
|-----------|------------------|
| Max dimension | Largest width/height allowed |
| Target shortest side | For aspect-ratio-aware scaling |
| Grayscale | Whether to strip color |
| Compression quality | JPEG quality level |
| Detail mode | "low" (85 tokens flat) vs "high" (tile-based) |
| Crop behavior | Whether to attempt smart cropping |

### Strategy logic by intent:

**Text Extraction**
- Keep resolution high enough for text legibility
- Grayscale is usually fine (text is text)
- Could crop to text regions if we detect them
- High detail mode (need the resolution)

**Scene Description**
- Moderate compression acceptable
- Keep color (matters for description)
- No cropping (need full context)
- High detail mode

**Object Detection**
- Can compress more aggressively
- Color optional depending on query
- No cropping
- High detail mode (but smaller image)

**Binary Questions**
- Maximum compression
- Grayscale often fine
- Low detail mode — this is where huge savings come from
- 85 tokens regardless of original size

**Detailed Analysis**
- Minimal compression
- Preserve everything
- High detail mode
- No aggressive optimizations

---

## GPT-4o Token Math

### Low Detail
Always 85 tokens. Period.

### High Detail
1. If image exceeds 2048px on any side, scale down to fit in 2048x2048 (preserve aspect ratio)
2. Scale so the shortest side is 768px
3. Count how many 512x512 tiles cover the image
4. Total = (tile_count × 170) + 85

### Examples
- 1024x1024 → scales to 768x768 → 4 tiles → 765 tokens
- 2048x4096 → scales to 1024x2048 → then 768x1536 → 6 tiles → 1105 tokens
- Any image in low detail → 85 tokens

---

## The Token Company Integration

Their API compresses text prompts. We compress images. Together = full multimodal compression.

**Their API:**
- Endpoint: `POST https://api.thetokencompany.com/v1/compress`
- Model: `bear-1`
- Key parameter: `aggressiveness` (0.0-1.0)
  - 0.1-0.3: Light compression
  - 0.4-0.6: Moderate
  - 0.7-0.9: Aggressive

**Integration story:**
1. Compress the text prompt with their API
2. Compress the image with our pipeline
3. Send both to vision model
4. Report combined savings

---

## Demo Requirements

The demo needs to make savings visceral and undeniable:

1. **Visual comparison** — show original vs optimized image
2. **Token counter** — real numbers, not estimates
3. **Cost projection** — "at 10K images/day, you'd save $X/month"
4. **Intent transparency** — show what intent was detected and why
5. **Multiple scenarios** — different query types showing different optimization paths

---

## Success Metrics

- Token reduction: 40-70% average across mixed workloads
- Accuracy preservation: Optimized responses should match unoptimized responses
- Latency: Preprocessing should add minimal overhead
- Demo clarity: Judges should immediately understand the value prop

---

## What We're NOT Building

- A replacement for vision models
- A general image compression tool
- Anything that requires training custom models
- A dashboard or analytics platform

We're building a smart preprocessing layer. That's it.

---

## Open Questions to Decide

1. What framework for the UI?
2. Zero-shot classifier vs keyword matching vs hybrid?
3. How to handle low-confidence intent classifications?
4. Should we support batch processing?
5. How aggressive should default compression be?