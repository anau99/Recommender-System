import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ENV_PATH = BASE_DIR / ".env"


if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
    print(f"Configuration loaded from: {ENV_PATH}")
else:
    print(f"Warning: .env file not found at {ENV_PATH}")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"