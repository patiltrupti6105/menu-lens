"""Build or replace the persistent ChromaDB dish index."""

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    PDF_PATH,
)
from indexing.data_loader import load_dishes
from indexing.document_builder import build_documents


def main() -> None:
    dishes = load_dishes(PDF_PATH)
    documents = build_documents(dishes)

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # PDF extraction and chunking finish before the previous index is replaced.
    existing = {item.name for item in client.list_collections()}
    if CHROMA_COLLECTION in existing:
        client.delete_collection(CHROMA_COLLECTION)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embedding_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=True,
    )
    print(f"Indexed {len(documents)} dishes in collection '{CHROMA_COLLECTION}'.")
    print(f"Persistent index: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
