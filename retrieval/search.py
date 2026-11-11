"""Search extracted menu text and return source-grounded chunks."""

from typing import Any

from retrieval.vector_store import get_resources


def search_menu(
    query: str, top_k: int = 5, filters: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    collection, embedder = get_resources()
    count = collection.count()
    if count == 0:
        return []
    vector = embedder.get_text_embedding(query)
    where_parts = []
    for key, value in (filters or {}).items():
        if value is None:
            continue
        if key == "max_price":
            where_parts.append({"price": {"$lte": value}})
        else:
            where_parts.append({key: {"$eq": value}})
    query_args = {}
    if len(where_parts) == 1:
        query_args["where"] = where_parts[0]
    elif where_parts:
        query_args["where"] = {"$and": where_parts}
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
        **query_args,
    )
    hits = []
    for index, text in enumerate(result["documents"][0] or []):
        metadata = (result.get("metadatas") or [[]])[0][index] or {}
        hits.append(
            {
                "id": (result.get("ids") or [[]])[0][index],
                "text": text,
                "source_file": metadata.get("source_file", "menu PDF"),
                "page_number": metadata.get("page_number"),
                "chunk_number": metadata.get("chunk_number"),
                "dish_id": metadata.get("dish_id"),
                "name": metadata.get("name"),
                "price": metadata.get("price"),
                "cuisine": metadata.get("cuisine"),
                "vegetarian": metadata.get("vegetarian"),
                "vegan": metadata.get("vegan"),
                "distance": (result.get("distances") or [[]])[0][index],
            }
        )
    return hits


def format_hit(hit: dict[str, Any]) -> str:
    page = hit.get("page_number")
    citation = hit.get("source_reference") or (
        f"{hit['source_file']}, page {page}" if page else hit["source_file"]
    )
    return f"[{hit['id']}] {hit['text']}\nSource: {citation}"
