"""Deterministic exact lookup, structured filtering, and semantic retrieval."""

from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any

from retrieval.vector_store import get_collection, get_resources


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _catalog() -> list[dict[str, Any]]:
    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    rows = []
    for i, dish_id in enumerate(result.get("ids", [])):
        metadata = (result.get("metadatas") or [])[i] or {}
        rows.append({"id": dish_id, "text": (result.get("documents") or [])[i] or "", **metadata})
    return rows


def _as_hit(row: dict[str, Any], distance: float | None = None) -> dict[str, Any]:
    return {
        "id": row.get("dish_id", row.get("id")), "dish_id": row.get("dish_id", row.get("id")),
        "text": row.get("text", ""), "source_file": "menu PDF", "page_number": row.get("page_number"),
        "name": row.get("name"), "description": row.get("description"), "price": row.get("price"),
        "cuisine": row.get("cuisine"), "category": row.get("category"),
        **{key: row.get(key) for key in ("vegetarian", "vegan", "jain", "gluten_free", "dairy_free", "egg_free")},
        "dietary_conditions": row.get("dietary_conditions", ""), "allergens": row.get("allergens"),
        "may_contain_allergens": row.get("may_contain_allergens"), "distance": distance,
    }


def list_dishes() -> list[dict[str, Any]]:
    return [_as_hit(row) for row in _catalog()]


def exact_lookup(query: str) -> list[dict[str, Any]]:
    """Resolve an ID, normalized exact name, then a very high-confidence typo."""
    q = query.strip()
    rows = _catalog()
    id_match = re.search(r"\bD\s*0*(\d{1,3})\b", q, re.I)
    if id_match:
        requested = f"D{int(id_match.group(1)):03d}"
        return [_as_hit(row) for row in rows if row.get("dish_id", row.get("id")) == requested]
    norm = normalize_name(q)
    names = [(row, normalize_name(row.get("name", ""))) for row in rows]
    exact = [row for row, name in names if name and (name == norm or name in norm)]
    if exact:
        exact.sort(key=lambda row: len(normalize_name(row.get("name", ""))), reverse=True)
        return [_as_hit(exact[0])]
    scores = sorted(((SequenceMatcher(None, norm, name).ratio(), row) for row, name in names), key=lambda x: x[0], reverse=True)
    if scores and scores[0][0] >= 0.93:
        return [_as_hit(scores[0][1])]
    return []


def resolve_dish_name(query: str) -> list[dict[str, Any]]:
    """Resolve an exact name or a unique set of words from one menu name.

    This is useful for short dish labels such as "ginger tea" for
    "Ginger Honey Tea". Ambiguous partial names deliberately remain unresolved.
    """
    exact = exact_lookup(query)
    if exact:
        return exact
    query_words = set(normalize_name(query).split())
    if len(query_words) < 2:
        return []
    matches = [
        row for row in _catalog()
        if query_words.issubset(set(normalize_name(row.get("name", "")).split()))
    ]
    return [_as_hit(matches[0])] if len(matches) == 1 else []


def filter_dishes(filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Apply structured constraints in Python; unknown dietary values fail positive filters."""
    filters = filters or {}
    results = []
    for hit in list_dishes():
        ok = True
        for key, value in filters.items():
            if value is None:
                continue
            if key == "max_price":
                ok &= isinstance(hit.get("price"), (int, float)) and hit["price"] <= value
            elif key in {"cuisine", "category"}:
                ok &= normalize_name(str(hit.get(key) or "")) == normalize_name(str(value))
            else:
                ok &= hit.get(key) is value
        if ok:
            results.append(hit)
    return results


def search_menu(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    collection, embedder = get_resources()
    count = collection.count()
    if not count:
        return []
    # Chroma's metadata filter applies first so approximate similarity cannot bypass constraints.
    where_parts = []
    for key, value in (filters or {}).items():
        if value is None:
            continue
        if key == "max_price":
            where_parts.append({"price": {"$lte": value}})
        elif key in {"vegetarian", "vegan", "jain", "gluten_free", "dairy_free", "egg_free"}:
            where_parts.append({key: {"$eq": bool(value)}})
        elif key in {"cuisine", "category"}:
            where_parts.append({key: {"$eq": str(value)}})
    args: dict[str, Any] = {}
    if len(where_parts) == 1:
        args["where"] = where_parts[0]
    elif where_parts:
        args["where"] = {"$and": where_parts}
    n_results = count if where_parts else min(top_k, count)
    result = collection.query(
        query_embeddings=[embedder.get_text_embedding(query)], n_results=n_results,
        include=["documents", "metadatas", "distances"], **args,
    )
    hits = []
    for i, text in enumerate(result["documents"][0] or []):
        metadata = (result.get("metadatas") or [[]])[0][i] or {}
        hits.append(_as_hit({"id": (result.get("ids") or [[]])[0][i], "text": text, **metadata},
                            (result.get("distances") or [[]])[0][i]))
    return hits


