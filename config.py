"""Central settings loaded from environment variables and the local .env file."""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = Path(os.getenv("MENULENS_PDF_PATH", PROJECT_ROOT / "data" / "MenuLens_Menu.pdf"))
CHROMA_PATH = Path(os.getenv("MENULENS_CHROMA_PATH", PROJECT_ROOT / "database" / "chroma_db"))
CHROMA_COLLECTION = os.getenv("MENULENS_CHROMA_COLLECTION", "menulens_dishes")
ACTIVE_COLLECTION_MANIFEST = CHROMA_PATH / "active_collection.json"
EMBEDDING_MODEL = os.getenv(
    "MENULENS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GEMINI_MODEL = os.getenv("MENULENS_GEMINI_MODEL", "gemini-3.7-flash")
GEMINI_FALLBACK_MODEL = os.getenv("MENULENS_GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
GEMINI_THINKING_LEVEL = os.getenv("MENULENS_GEMINI_THINKING_LEVEL", "low")
GEMINI_REQUEST_TIMEOUT = float(os.getenv("MENULENS_GEMINI_REQUEST_TIMEOUT", "30"))
GRADIO_SHARE = os.getenv("MENULENS_GRADIO_SHARE", "false").strip().lower() in {"1", "true", "yes", "on"}
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

