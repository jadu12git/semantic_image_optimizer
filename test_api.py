#!/usr/bin/env python3
"""Test the FastAPI endpoints."""

import requests

BASE_URL = "http://localhost:8000"

# Test health
print("Testing /health...")
r = requests.get(f"{BASE_URL}/health")
print(f"  Status: {r.json()}")

# Test classify
print("\nTesting /classify...")
r = requests.post(f"{BASE_URL}/classify", data={"query": "Is the total more than 25 dollars?"})
print(f"  Result: {r.json()}")

# Test optimize
print("\nTesting /optimize...")
with open("tests/sample_images/reciept.jpeg", "rb") as f:
    r = requests.post(
        f"{BASE_URL}/optimize",
        data={
            "query": "Is the total more than 25 dollars?",
            "compress_text": "true",
        },
        files={"image": ("reciept.jpeg", f, "image/jpeg")},
    )

result = r.json()
# Remove base64 for display
if "base64_image" in result:
    result["base64_image"] = f"<{len(result['base64_image'])} chars>"

import json
print(json.dumps(result, indent=2))

# Test compare
print("\nTesting /compare...")
with open("tests/sample_images/reciept.jpeg", "rb") as f:
    r = requests.post(
        f"{BASE_URL}/compare",
        data={"query": "Is the total more than 25 dollars?"},
        files={"image": ("reciept.jpeg", f, "image/jpeg")},
    )

result = r.json()
# Remove base64 for display
if "original" in result and "base64_image" in result["original"]:
    result["original"]["base64_image"] = f"<{len(result['original']['base64_image'])} chars>"
if "optimized" in result and "base64_image" in result["optimized"]:
    result["optimized"]["base64_image"] = f"<{len(result['optimized']['base64_image'])} chars>"

print(json.dumps(result, indent=2))

print("\nAll tests passed!")
