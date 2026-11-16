"""Build and safely activate a new versioned ChromaDB dish collection."""

import json
import os
import tempfile
from datetime import datetime, timezone

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import ACTIVE_COLLECTION_MANIFEST, CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL, PDF_PATH
from indexing.data_loader import load_dishes, load_menu_disclaimer
from indexing.document_builder import build_documents


def _activate_manifest(data: dict) -> None:
    """Replace the active manifest atomically after the collection has passed checks."""
    ACTIVE_COLLECTION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix="active_collection_", suffix=".tmp", dir=CHROMA_PATH)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, ACTIVE_COLLECTION_MANIFEST)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def build_index() -> str:
    dishes = load_dishes(PDF_PATH)
    documents = build_documents(dishes)
    expected_ids = {dish.dish_id for dish in dishes}

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    collection_name = f"{CHROMA_COLLECTION}_{version}"
    collection = client.create_collection(collection_name)
    try:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        context = StorageContext.from_defaults(vector_store=vector_store)
        embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        VectorStoreIndex.from_documents(
            documents, storage_context=context, embed_model=embedder, show_progress=True,
        )
        # Check both total count and IDs before the application is pointed at this index.
        if collection.count() != len(expected_ids):
            raise RuntimeError(f"Index validation failed: expected {len(expected_ids)} records, got {collection.count()}.")
        # LlamaIndex/Chroma may assign UUIDs as storage IDs. Dish IDs live in
        # the document metadata and are the stable IDs the application uses.
        indexed_records = collection.get(include=["metadatas"])
        indexed_ids = {
            metadata.get("dish_id")
            for metadata in indexed_records.get("metadatas", [])
            if metadata and metadata.get("dish_id")
        }
        if indexed_ids != expected_ids:
            missing = sorted(expected_ids - indexed_ids)
            extra = sorted(indexed_ids - expected_ids)
            raise RuntimeError(f"Index validation failed; missing IDs {missing}, unexpected IDs {extra}.")
        _activate_manifest({
            "collection": collection_name, "base_collection": CHROMA_COLLECTION,
            "document_count": len(expected_ids), "dish_ids": sorted(expected_ids),
            "menu_disclaimer": load_menu_disclaimer(PDF_PATH), "created_at": version,
        })
    except BaseException:
        # Remove only this unactivated build attempt; the active and rollback collections stay untouched.
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        raise
    print(f"Validated and activated {len(documents)} dishes in collection '{collection_name}'.")
    print(f"Persistent index: {CHROMA_PATH}")
    print("Restart a running MenuLens app to load the new collection; its vector store is cached in memory.")
    return collection_name


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
