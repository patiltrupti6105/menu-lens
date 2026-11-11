"""Small helpers for the app's user-facing setup messages."""

from config import CHROMA_COLLECTION, CHROMA_PATH


def index_ready() -> bool:
    if not CHROMA_PATH.exists():
        return False
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        names = {collection.name for collection in client.list_collections()}
        return CHROMA_COLLECTION in names
    except Exception:
        return False
