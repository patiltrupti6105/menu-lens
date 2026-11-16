"""Small helpers for the app's setup and status messages."""

from config import CHROMA_PATH


def index_ready() -> bool:
    if not CHROMA_PATH.exists():
        return False
    try:
        from retrieval.vector_store import active_collection_name
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_collection(active_collection_name())
        return collection.count() > 0
    except Exception:
        return False
