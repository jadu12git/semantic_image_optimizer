"""
Token Company API Integration

Compresses text prompts using Token Company's bear-1 model.
Combined with our image compression, this provides full multimodal optimization.

API Docs: https://thetokencompany.com/docs
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use env vars directly


# API Configuration
TOKEN_COMPANY_API_URL = "https://api.thetokencompany.com/v1/compress"
TOKEN_COMPANY_MODEL = "bear-1"


@dataclass
class CompressionResult:
    """Result of text compression"""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    savings_percent: float
    compression_time: float
    aggressiveness: float


class TokenCompanyClient:
    """
    Client for Token Company's text compression API.

    Compresses text prompts while preserving semantic meaning.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.

        Args:
            api_key: Token Company API key. If not provided, reads from
                     TOKEN_COMPANY_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("TOKEN_COMPANY_API_KEY")
        if not self.api_key:
            print("Warning: No Token Company API key provided. Text compression will be simulated.")

    def compress(
        self,
        text: str,
        aggressiveness: float = 0.5,
        max_output_tokens: Optional[int] = None,
        min_output_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """
        Compress text using Token Company's bear-1 model.

        Args:
            text: The text to compress
            aggressiveness: 0.0-1.0, higher = more compression
                - 0.1-0.3: Light compression
                - 0.4-0.6: Moderate compression (default)
                - 0.7-0.9: Aggressive compression
            max_output_tokens: Optional upper limit on output tokens
            min_output_tokens: Optional lower limit on output tokens

        Returns:
            CompressionResult with original and compressed text
        """
        # If no API key, simulate compression
        if not self.api_key:
            return self._simulate_compression(text, aggressiveness)

        # Build request
        payload = {
            "model": TOKEN_COMPANY_MODEL,
            "input": text,
            "compression_settings": {
                "aggressiveness": aggressiveness,
            }
        }

        if max_output_tokens is not None:
            payload["compression_settings"]["max_output_tokens"] = max_output_tokens
        if min_output_tokens is not None:
            payload["compression_settings"]["min_output_tokens"] = min_output_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                TOKEN_COMPANY_API_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            original_tokens = data["original_input_tokens"]
            compressed_tokens = data["output_tokens"]
            tokens_saved = original_tokens - compressed_tokens
            savings_percent = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0

            return CompressionResult(
                original_text=text,
                compressed_text=data["output"],
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                tokens_saved=tokens_saved,
                savings_percent=savings_percent,
                compression_time=data.get("compression_time", 0),
                aggressiveness=aggressiveness,
            )

        except requests.exceptions.RequestException as e:
            print(f"Token Company API error: {e}")
            # Fall back to simulation on error
            return self._simulate_compression(text, aggressiveness)

    def _simulate_compression(self, text: str, aggressiveness: float) -> CompressionResult:
        """
        Simulate compression when API is not available.

        This provides a rough approximation for demo/testing purposes.
        Real compression would be done by Token Company's API.
        """
        # Rough token estimate (words * 1.3)
        words = text.split()
        original_tokens = int(len(words) * 1.3)

        # Simulate compression based on aggressiveness
        # Higher aggressiveness = more tokens removed
        compression_ratio = 1 - (aggressiveness * 0.6)  # 0.6 max removal
        compressed_tokens = max(1, int(original_tokens * compression_ratio))

        # Create simulated compressed text (just truncate for demo)
        # Real API would do semantic compression
        target_words = max(1, int(len(words) * compression_ratio))

        # Simple heuristic: keep first and last parts, remove middle
        if len(words) > target_words:
            keep_start = target_words // 2
            keep_end = target_words - keep_start
            compressed_words = words[:keep_start] + ["..."] + words[-keep_end:]
            compressed_text = " ".join(compressed_words)
        else:
            compressed_text = text

        tokens_saved = original_tokens - compressed_tokens
        savings_percent = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text + " [SIMULATED]",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            savings_percent=savings_percent,
            compression_time=0.0,
            aggressiveness=aggressiveness,
        )


# Singleton client
_client_instance: Optional[TokenCompanyClient] = None


def get_client(api_key: Optional[str] = None) -> TokenCompanyClient:
    """Get or create the singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = TokenCompanyClient(api_key)
    return _client_instance


def compress_text(
    text: str,
    aggressiveness: float = 0.5,
    api_key: Optional[str] = None,
) -> CompressionResult:
    """
    Convenience function to compress text.

    Args:
        text: Text to compress
        aggressiveness: Compression level (0.0-1.0)
        api_key: Optional API key (uses env var if not provided)

    Returns:
        CompressionResult
    """
    client = get_client(api_key)
    return client.compress(text, aggressiveness)


if __name__ == "__main__":
    # Test the client
    test_text = """
    I went to dinner last night at this Italian restaurant and I'm trying to remember
    if I was overcharged. I think the total should have been around $45 but I'm not
    sure. Can you look at this receipt image and tell me if there's a tip already
    included? I just need a yes or no answer because I want to know if I should
    dispute the charge with my credit card company. The receipt might be a bit blurry
    but hopefully you can still make it out.
    """

    print("Token Company Text Compression Test")
    print("=" * 60)

    # Test with different aggressiveness levels
    for level in [0.3, 0.5, 0.7]:
        result = compress_text(test_text.strip(), aggressiveness=level)

        print(f"\nAggressiveness: {level}")
        print(f"  Original tokens:   {result.original_tokens}")
        print(f"  Compressed tokens: {result.compressed_tokens}")
        print(f"  Tokens saved:      {result.tokens_saved} ({result.savings_percent:.1f}%)")
        print(f"  Compressed text:   {result.compressed_text[:100]}...")
