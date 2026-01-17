# semantic_image_optimizer
optimize image input for LLMs to save on tokens but maintain performance
file structure
token-squeeze/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── intent.py        # Intent classification
│   │   ├── strategies.py    # Compression strategies per intent
│   │   └── processor.py     # Pillow image operations
│   ├── api/
│   │   └── vision.py        # GPT-4o wrapper + token counting
│   └── utils/
│       └── tokens.py        # Token calculation helpers
├── frontend/
│   └── app.py               # Streamlit app (or React folder)
├── tests/
│   └── sample_images/       # Test receipts, photos, etc.
├── requirements.txt
├── .env.example
└── README.md