"""
Intent Classifier for TokenSqueeze (Lite Version for Serverless)

Keyword-only approach for Vercel deployment (no PyTorch/Transformers).
Falls back to SCENE_DESCRIPTION for unmatched queries.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    """Supported intent categories"""
    BINARY_QUESTION = "BINARY_QUESTION"
    TEXT_EXTRACTION = "TEXT_EXTRACTION"
    SCENE_DESCRIPTION = "SCENE_DESCRIPTION"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    DETAILED_ANALYSIS = "DETAILED_ANALYSIS"
    COMPARISON = "COMPARISON"


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: Intent
    confidence: float
    all_scores: Dict[str, float]
    recommended_strategy: Dict
    method: str  # "keyword" or "fallback"


# Keyword patterns for each intent (checked first)
KEYWORD_PATTERNS = {
    Intent.BINARY_QUESTION: [
        r"^(is|are|does|do|can|will|was|were|has|have|did)\s+.+\?$",
        r"^(is|are|does|do|can|will|was|were|has|have|did)\s+(the|this|it|there)",
        r"yes\s*(or|\/)\s*no",
        r"true\s*(or|\/)\s*false",
        r"^is\s+(this|there|it)\s+",
        r"^does\s+(this|it)\s+",
        r"^are\s+there\s+",
        r"\b(correct|right|accurate|valid|true|real|legit|legitimate)\b.*\?",
        r"\bis\s+(the\s+)?(total|amount|price|sum|cost|bill)\b",
        r"\bmore\s+than\b|\bless\s+than\b|\bgreater\s+than\b|\bunder\b|\bover\b",
        r"\b(answer|reply|respond)\s+(yes|no)\b",
        r"\bif\s+(the\s+)?(total|amount|price|sum|cost|bill)\s+(is\s+)?(correct|right|accurate|real)\b",
        r"\bwant\s+to\s+know\s+if\b",
        r"\b(legally|legal|allowed|permitted|can i|am i)\b",
    ],
    Intent.TEXT_EXTRACTION: [
        r"\b(ocr|transcribe|extract\s+text|read\s+(the\s+)?text)\b",
        r"\bwhat\s+(does|do)\s+(it|the\s+\w+)\s+say\b",
        r"\bread\s+(this|the|what)",
        r"\bextract\s+(all\s+)?(the\s+)?(text|words)",
        r"\bwhat\s+(is|are)\s+written\b",
        r"\btranscribe\b",
    ],
    Intent.OBJECT_DETECTION: [
        r"\bhow\s+many\b",
        r"\bcount\s+(the|all|how)\b",
        r"\bfind\s+(all|the|and\s+count)\b",
        r"\blocate\s+(all|the)\b",
        r"\bnumber\s+of\b",
    ],
    Intent.DETAILED_ANALYSIS: [
        r"\b(chart|graph|diagram|infographic|visualization)\b",
        r"\banalyze\s+(this|the)\b",
        r"\bexplain\s+(the\s+)?(data|trend|pattern)\b",
        r"\bwhat\s+does\s+(this|the)\s+(chart|graph|diagram)\s+show\b",
    ],
    Intent.COMPARISON: [
        r"\bcompare\b",
        r"\bdifference(s)?\s+between\b",
        r"\bspot\s+(the\s+)?difference",
        r"\bwhich\s+(one|image|photo)\s+(is|has|looks)\b",
        r"\bsame\s+or\s+different\b",
    ],
    Intent.SCENE_DESCRIPTION: [
        r"^describe\s+(this|the|what)",
        r"\bwhat\s+(is|are)\s+(in|shown|happening|going\s+on)\b",
        r"\btell\s+me\s+about\b",
        r"\bexplain\s+(this|what)\s+(image|picture|photo)",
    ],
}

# Optimization strategies for each intent
STRATEGIES = {
    Intent.BINARY_QUESTION: {
        "detail_mode": "low",
        "max_dimension": 512,
        "grayscale": True,
        "jpeg_quality": 60,
        "smart_crop": True,
        "description": "Maximum compression - simple yes/no doesn't need detail"
    },
    Intent.TEXT_EXTRACTION: {
        "detail_mode": "high",
        "max_dimension": 2048,
        "grayscale": True,
        "jpeg_quality": 85,
        "smart_crop": True,
        "description": "High resolution for text legibility, grayscale OK"
    },
    Intent.SCENE_DESCRIPTION: {
        "detail_mode": "high",
        "max_dimension": 1024,
        "grayscale": False,
        "jpeg_quality": 75,
        "description": "Moderate compression, preserve colors"
    },
    Intent.OBJECT_DETECTION: {
        "detail_mode": "high",
        "max_dimension": 1024,
        "grayscale": False,
        "jpeg_quality": 70,
        "description": "Need enough detail to identify and count objects"
    },
    Intent.DETAILED_ANALYSIS: {
        "detail_mode": "high",
        "max_dimension": 2048,
        "grayscale": False,
        "jpeg_quality": 90,
        "description": "Minimal compression - preserve all details"
    },
    Intent.COMPARISON: {
        "detail_mode": "high",
        "max_dimension": 1024,
        "grayscale": False,
        "jpeg_quality": 80,
        "description": "Consistent processing for fair comparison"
    },
}


class IntentClassifier:
    """
    Lightweight intent classifier using keyword matching only.
    For serverless deployment where PyTorch is too heavy.
    """

    def __init__(self):
        print("Loading lite intent classifier (keyword-only)...")
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in KEYWORD_PATTERNS.items()
        }
        print("Lite intent classifier ready.")

    def _keyword_match(self, query: str) -> Optional[Tuple[Intent, float]]:
        """Try to match query against keyword patterns."""
        query_lower = query.lower().strip()

        matches = []
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    matches.append(intent)
                    break

        if len(matches) == 1:
            return (matches[0], 0.95)
        elif len(matches) > 1:
            # Multiple matches - prefer BINARY_QUESTION if present (most savings)
            if Intent.BINARY_QUESTION in matches:
                return (Intent.BINARY_QUESTION, 0.85)
            return (matches[0], 0.75)
        return None

    def classify(self, query: str) -> IntentResult:
        """Classify a user query into an intent category."""
        keyword_result = self._keyword_match(query)

        if keyword_result:
            intent, confidence = keyword_result
            method = "keyword"
        else:
            # Fallback to SCENE_DESCRIPTION (safe default)
            intent = Intent.SCENE_DESCRIPTION
            confidence = 0.5
            method = "fallback"

        all_scores = {i.value: 0.0 for i in Intent}
        all_scores[intent.value] = confidence
        strategy = STRATEGIES[intent].copy()

        return IntentResult(
            intent=intent,
            confidence=confidence,
            all_scores=all_scores,
            recommended_strategy=strategy,
            method=method
        )

    def classify_batch(self, queries: List[str]) -> List[IntentResult]:
        """Classify multiple queries."""
        return [self.classify(query) for query in queries]


# Singleton instance
_classifier_instance: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


def classify_intent(query: str) -> IntentResult:
    """Convenience function to classify intent."""
    classifier = get_classifier()
    return classifier.classify(query)


if __name__ == "__main__":
    classifier = IntentClassifier()

    test_queries = [
        "Is the total more than $25?",
        "Is this amount correct?",
        "Does the stitching look legitimate?",
        "Am I legally parked right now?",
        "Extract the text from this image",
        "How many people are in this photo?",
        "Describe what you see",
    ]

    print("\nLITE INTENT CLASSIFICATION (Keywords only)")
    print("=" * 60)
    for query in test_queries:
        result = classifier.classify(query)
        print(f"{result.intent.value:20} ({result.confidence:.0%}) [{result.method:8}] <- \"{query}\"")
