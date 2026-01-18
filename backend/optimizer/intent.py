"""
Intent Classifier for TokenSqueeze

Hybrid approach:
1. Keyword matching for high-confidence cases (fast, accurate)
2. Zero-shot classification for ambiguous cases (DeBERTa, 200M params)

This determines the optimal image compression strategy.
"""

import re
import torch
from transformers import pipeline
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
    method: str  # "keyword" or "zero-shot"


# Keyword patterns for each intent (checked first)
KEYWORD_PATTERNS = {
    Intent.BINARY_QUESTION: [
        r"^(is|are|does|do|can|will|was|were|has|have|did)\s+.+\?$",  # Questions starting with be/do verbs
        r"^(is|are|does|do|can|will|was|were|has|have|did)\s+(the|this|it|there)",  # Even without ? at end
        r"yes\s*(or|\/)\s*no",
        r"true\s*(or|\/)\s*false",
        r"^is\s+(this|there|it)\s+",
        r"^does\s+(this|it)\s+",
        r"^are\s+there\s+",
        r"\b(correct|right|accurate|valid|true|real|legit|legitimate)\b.*\?",  # Verification questions
        r"\bis\s+(the\s+)?(total|amount|price|sum|cost|bill)\b",  # Questions about totals/amounts
        r"\bmore\s+than\b|\bless\s+than\b|\bgreater\s+than\b|\bunder\b|\bover\b",  # Comparison questions
        r"\b(answer|reply|respond)\s+(yes|no)\b",  # Explicit yes/no request
        r"\bif\s+(the\s+)?(total|amount|price|sum|cost|bill)\s+(is\s+)?(correct|right|accurate|real)\b",  # "if the total is correct"
        r"\bwant\s+to\s+know\s+if\b",  # "want to know if" questions are usually binary
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

# Zero-shot labels for fallback
INTENT_LABELS = {
    Intent.BINARY_QUESTION: "a simple yes or no question",
    Intent.TEXT_EXTRACTION: "a request to read or transcribe text",
    Intent.SCENE_DESCRIPTION: "a request to describe what is in the image",
    Intent.OBJECT_DETECTION: "a request to count or locate objects",
    Intent.DETAILED_ANALYSIS: "a request to analyze a chart or diagram",
    Intent.COMPARISON: "a request to compare images",
}

# Optimization strategies for each intent
STRATEGIES = {
    Intent.BINARY_QUESTION: {
        "detail_mode": "low",       # 85 tokens flat - huge savings!
        "max_dimension": 512,
        "grayscale": True,
        "jpeg_quality": 60,
        "smart_crop": True,         # Crop document/receipt to save even more tokens
        "description": "Maximum compression - simple yes/no doesn't need detail"
    },
    Intent.TEXT_EXTRACTION: {
        "detail_mode": "high",
        "max_dimension": 2048,
        "grayscale": True,          # Text is readable in grayscale
        "jpeg_quality": 85,
        "smart_crop": True,         # Crop to document bounds for focus
        "description": "High resolution for text legibility, grayscale OK"
    },
    Intent.SCENE_DESCRIPTION: {
        "detail_mode": "high",
        "max_dimension": 1024,
        "grayscale": False,         # Color matters for description
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
    Hybrid intent classifier.

    1. First tries keyword matching (fast, deterministic)
    2. Falls back to zero-shot classification if no keyword match
    """

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        """
        Initialize the classifier.

        Args:
            model_name: HuggingFace model for zero-shot classification fallback
        """
        print(f"Loading intent classifier...")

        # Compile regex patterns
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in KEYWORD_PATTERNS.items()
        }

        # Lazy load zero-shot model (only when needed)
        self._zero_shot_classifier = None
        self._model_name = model_name

        print("Intent classifier ready.")

    def _get_zero_shot_classifier(self):
        """Lazy load the zero-shot classifier."""
        if self._zero_shot_classifier is None:
            print("Loading zero-shot model (first use)...")
            device = 0 if torch.cuda.is_available() else -1
            self._zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=device
            )
        return self._zero_shot_classifier

    def _keyword_match(self, query: str) -> Optional[Tuple[Intent, float]]:
        """
        Try to match query against keyword patterns.

        Returns:
            Tuple of (Intent, confidence) if matched, None otherwise
        """
        query_lower = query.lower().strip()

        matches = []
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    matches.append(intent)
                    break

        if len(matches) == 1:
            # Single clear match
            return (matches[0], 0.95)
        elif len(matches) > 1:
            # Multiple matches - ambiguous, let zero-shot decide
            return None
        else:
            # No matches
            return None

    def _zero_shot_classify(self, query: str) -> Tuple[Intent, float, Dict[str, float]]:
        """
        Use zero-shot classification for ambiguous queries.
        """
        classifier = self._get_zero_shot_classifier()
        labels = list(INTENT_LABELS.values())
        label_to_intent = {v: k for k, v in INTENT_LABELS.items()}

        result = classifier(query, candidate_labels=labels, multi_label=False)

        all_scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            intent = label_to_intent[label]
            all_scores[intent.value] = score

        best_label = result["labels"][0]
        best_intent = label_to_intent[best_label]
        best_score = result["scores"][0]

        return (best_intent, best_score, all_scores)

    def classify(self, query: str) -> IntentResult:
        """
        Classify a user query into an intent category.

        Args:
            query: The user's query text

        Returns:
            IntentResult with intent, confidence, scores, and recommended strategy
        """
        # Try keyword matching first
        keyword_result = self._keyword_match(query)

        if keyword_result:
            intent, confidence = keyword_result
            # Create scores dict with high score for matched intent
            all_scores = {i.value: 0.0 for i in Intent}
            all_scores[intent.value] = confidence
            method = "keyword"
        else:
            # Fall back to zero-shot
            intent, confidence, all_scores = self._zero_shot_classify(query)
            method = "zero-shot"

        # Get recommended strategy
        strategy = STRATEGIES[intent].copy()

        return IntentResult(
            intent=intent,
            confidence=confidence,
            all_scores=all_scores,
            recommended_strategy=strategy,
            method=method
        )

    def classify_batch(self, queries: List[str]) -> List[IntentResult]:
        """
        Classify multiple queries efficiently.

        Args:
            queries: List of query strings

        Returns:
            List of IntentResult objects
        """
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
    """
    Convenience function to classify intent.

    Args:
        query: User query text

    Returns:
        IntentResult
    """
    classifier = get_classifier()
    return classifier.classify(query)


if __name__ == "__main__":
    # Test the classifier
    classifier = IntentClassifier()

    test_queries = [
        # Binary questions
        "Is there a dog in this image?",
        "Yes or no, is this a cat?",
        "Does this image contain food?",
        "Are there any people in this photo?",
        "Is the total correct?",
        "Is the total on this the real amount or not",
        "I ordered food from this restaurant and I want to know if the total is correct",
        "Is the total more than $25?",
        "Is this amount accurate?",

        # Object detection / counting
        "How many boxes of chalk are in this image?",
        "How many people are inside this image?",
        "Count the cars in this photo",
        "Find all the animals in this picture",

        # Text extraction
        "Extract the text from this image for me",
        "Read the receipt and tell me the total",
        "OCR this document",
        "What does the sign say?",

        # Scene description
        "Describe what you see in this image",
        "What is happening in this photo?",
        "Tell me about this picture",

        # Detailed analysis
        "What does this chart show?",
        "Analyze this graph and explain the trends",

        # Comparison
        "Compare these two images",
        "What's the difference between these pictures?",
    ]

    print("\n" + "="*75)
    print("HYBRID INTENT CLASSIFICATION (Keywords + DeBERTa fallback)")
    print("="*75 + "\n")

    for query in test_queries:
        result = classifier.classify(query)
        method_tag = f"[{result.method}]"
        print(f"{result.intent.value:20} ({result.confidence:5.1%}) {method_tag:12} ← \"{query}\"")
