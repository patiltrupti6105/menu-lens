"""Shared persistent ChromaDB and embedding model access."""

import os

import chromadb

from config import CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL

_client = None
_collection = None
_embedder = None


def get_resources():
    global _client, _collection, _embedder
    if _collection is None:
        # Load the transformer stack only when a search actually needs embeddings.
        # The index build already downloaded this model, so app requests can use cache only.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        if not CHROMA_PATH.exists():
            raise RuntimeError("Menu index not found. Build it with: python -m indexing.build_index")
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            _collection = _client.get_collection(CHROMA_COLLECTION)
        except Exception as exc:
            raise RuntimeError("Menu index is missing. Rebuild it with: python -m indexing.build_index") from exc
        _embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    return _collection, _embedder


def warm_up() -> None:
    """Load Chroma and the cached embedding model before serving chat requests."""
    get_resources()
