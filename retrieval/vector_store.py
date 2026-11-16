"""Shared persistent ChromaDB and embedding model access."""

import json
import os

import chromadb

from config import ACTIVE_COLLECTION_MANIFEST, CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL

_client = None
_collection = None
_embedder = None
_manifest: dict = {}


def active_collection_name() -> str:
    """Read the atomic active manifest, falling back to the legacy collection name."""
    if ACTIVE_COLLECTION_MANIFEST.is_file():
        try:
            with ACTIVE_COLLECTION_MANIFEST.open(encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data.get("collection"), str) and data["collection"]:
                return data["collection"]
        except (OSError, ValueError, TypeError):
            raise RuntimeError("The active menu index manifest is invalid; rebuild the index.")
    return CHROMA_COLLECTION


def get_collection():
    """Load Chroma and its manifest without initializing the embedding model."""
    global _client, _collection, _manifest
    if _collection is None:
        if not CHROMA_PATH.exists():
            raise RuntimeError("Menu index not found. Build it with: python -m indexing.build_index")
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        name = active_collection_name()
        try:
            _collection = _client.get_collection(name)
            if ACTIVE_COLLECTION_MANIFEST.is_file():
                with ACTIVE_COLLECTION_MANIFEST.open(encoding="utf-8") as file:
                    _manifest = json.load(file)
        except Exception as exc:
            raise RuntimeError("Menu index is missing. Rebuild it with: python -m indexing.build_index") from exc
    return _collection


def get_resources():
    global _embedder
    collection = get_collection()
    if _embedder is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        _embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    return collection, _embedder


def get_menu_manifest() -> dict:
    get_collection()
    return dict(_manifest)


def reset_cached_resources() -> None:
    """Used by local verification after switching an index in the same process."""
    global _client, _collection, _embedder, _manifest
    _client = _collection = _embedder = None
    _manifest = {}


def warm_up() -> None:
    get_resources()
