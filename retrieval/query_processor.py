"""Predictable query routing, grounded retrieval, and direct Gemini response generation."""

import re
from typing import Any

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODEL, GEMINI_MODEL, GEMINI_REQUEST_TIMEOUT, GEMINI_THINKING_LEVEL
from retrieval.search import exact_lookup, list_dishes, search_menu
from utils.citations import menu_page_citation
from indexing.data_loader import MENU_ALLERGEN_DISCLAIMER
from retrieval.vector_store import get_menu_manifest


CATEGORY_ALIASES = {
    "tea": "Teas & Iced Teas", "teas": "Teas & Iced Teas",
    "iced tea": "Teas & Iced Teas", "iced teas": "Teas & Iced Teas",
    "coffee": "Coffee & Hot Drinks", "hot drinks": "Coffee & Hot Drinks",
    "soup": "Soups", "soups": "Soups", "salad": "Salads", "salads": "Salads",
    "starter": "Starters & Tandoor", "starters": "Starters & Tandoor",
    "pizza": "Wood-Fired Pizza", "pizzas": "Wood-Fired Pizza",
    "pasta": "Pasta & Baked Classics", "pastas": "Pasta & Baked Classics",
    "biryani": "Biryani & Rice", "breads": "Breads from the Tandoor",
}


def _plain(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        return "\n".join(x.get("text", "") for x in item if isinstance(x, dict))
    return str(item or "")


def _history_text(history: list[dict], limit: int = 8) -> str:
    lines = []
    for item in history[-limit:]:
        role = item.get("role", "")
        content = _plain(item.get("content", item.get("text", ""))).strip()
        if role == "assistant":
            content = re.split(r"\n\s*\*\*Menu sources\*\*", content, maxsplit=1, flags=re.I)[0]
            content = re.sub(r"(?im)^\s*(?:[-*]\s*)?Source:.*$", "", content).strip()
        if role in {"user", "assistant"} and content:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(lines)


def _filters(question: str) -> dict[str, Any]:
    q = question.casefold()
    result: dict[str, Any] = {}
    for key, words in {
        "vegetarian": (r"\bvegetarian\b",), "vegan": (r"\bvegan\b",),
        "jain": (r"\bjain(?:-friendly)?\b",), "gluten_free": (r"\bgluten[- ]free\b",),
        "dairy_free": (r"\bdairy[- ]free\b",), "egg_free": (r"\begg[- ]free\b",),
    }.items():
        if any(re.search(word, q) for word in words):
            result[key] = True
    price = re.search(r"\b(?:under|below|less than|at most|no more than|up to|within)\s*(?:₹|rs\.?\s*)?([\d,]+)", q)
    if not price:
        price = re.search(r"(?:₹|rs\.?\s*)([\d,]+)\s*(?:or less|maximum|max)?\b", q)
    if price:
        result["max_price"] = float(price.group(1).replace(",", ""))

    # These values come from the explicit PDF section headers.
    category_names = sorted({hit.get("category") for hit in list_dishes() if hit.get("category")}, key=len, reverse=True)
    cuisine_names = sorted({hit.get("cuisine") for hit in list_dishes() if hit.get("cuisine")}, key=len, reverse=True)
    for field, candidates in (("category", category_names), ("cuisine", cuisine_names)):
        selected = next((value for value in candidates if value.casefold() in q), None)
        if selected:
            result[field] = selected
    for alias, category in sorted(CATEGORY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            result.setdefault("category", category)
            break
    return result


def _evidence_text(hits: list[dict]) -> str:
    """Present retrieved dish fields explicitly so Gemini can distinguish unknowns."""
    records = []
    for hit in hits:
        fields = [f"Dish ID: {hit.get('dish_id') or 'not specified'}", f"Name: {hit.get('name') or 'not specified'}"]
        for key, label in (("description", "Description"), ("price", "Price"), ("cuisine", "Cuisine"), ("category", "Category")):
            value = hit.get(key)
            if value is not None:
                fields.append(f"{label}: {'₹' + format(value, 'g') if key == 'price' else value}")
        for key in ("vegetarian", "vegan", "jain", "gluten_free", "dairy_free", "egg_free"):
            if hit.get(key) is not None:
                fields.append(f"{key.replace('_', ' ').title()}: {hit[key]}")
        for key, label in (("dietary_conditions", "Dietary conditions"), ("allergens", "Confirmed allergens"), ("may_contain_allergens", "May contain allergens")):
            value = hit.get(key)
            if value:
                fields.append(f"{label}: {value}")
        if hit.get("page_number"):
            fields.append(f"PDF page: {hit['page_number']}")
        if hit.get("text"):
            fields.append(f"Menu record text:\n{hit['text']}")
        records.append("\n".join(fields))
    return "\n\n---\n\n".join(records)


def _gemini_answer(question: str, history: str, evidence: str, system: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Google API key is not configured")
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    candidates = list(dict.fromkeys([GEMINI_MODEL, GEMINI_FALLBACK_MODEL]))
    error = None
    for i, model in enumerate(candidates):
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"Recent conversation:\n{history or '(none)'}\n\nUser request: {question}\n\nRetrieved menu records:\n{evidence}",
                config={
                    "system_instruction": system,
                    "max_output_tokens": 1400,
                    "thinking_config": {"thinking_level": GEMINI_THINKING_LEVEL},
                    "http_options": {"timeout": int(GEMINI_REQUEST_TIMEOUT * 1000)},
                },
            )
            answer = getattr(response, "text", None)
            if answer and answer.strip():
                return answer.strip()
            raise RuntimeError("Gemini returned an empty response")
        except Exception as exc:
            error = exc
            text = str(exc).lower()
            temporary = any(code in text for code in ("429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "timeout", "deadline_exceeded"))
            if i + 1 >= len(candidates) or not temporary:
                break
    raise error


def _remove_generated_sources(answer: str) -> str:
    """Remove model-generated citations; the application adds verified ones beside dishes."""
    answer = re.split(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:menu\s+)?sources?(?:\*\*)?:?\s*$",
        answer, maxsplit=1,
    )[0]
    answer = re.sub(r"(?im)^\s*(?:[-*]\s*)?Source:.*$", "", answer)
    return answer.rstrip()


def _cite_selected_dishes(answer: str, hits: list[dict]) -> str:
    """Put citations beside dishes the answer actually selected, not every retrieved hit."""
    mentioned_ids = {value.upper() for value in re.findall(r"\bD\d{3}\b", answer, re.I)}
    selected = [hit for hit in hits if hit.get("dish_id", "").upper() in mentioned_ids]
    if not selected:
        selected = [
            hit for hit in hits
            if hit.get("name") and hit["name"].casefold() in answer.casefold()
        ]
    lines = answer.splitlines()
    cited: set[str] = set()
    for index, line in enumerate(lines):
        links = []
        for hit in selected:
            dish_id = hit.get("dish_id", "")
            if dish_id in cited:
                continue
            has_id = bool(dish_id and re.search(rf"\b{re.escape(dish_id)}\b", line, re.I))
            has_name = bool(hit.get("name") and hit["name"].casefold() in line.casefold())
            if has_id or has_name:
                links.append(menu_page_citation(hit.get("page_number"), hit.get("name")))
                cited.add(dish_id)
        if links:
            lines[index] = line.rstrip() + " — Source: " + "; ".join(links)
    return "\n".join(lines).rstrip()


def _allergen_warning(answer: str) -> str:
    disclaimer = get_menu_manifest().get("menu_disclaimer") or MENU_ALLERGEN_DISCLAIMER
    return answer + "\n\nAllergen reminder: " + disclaimer


def _fallback(hits: list[dict], reason: str = "") -> str:
    if not hits:
        return "I couldn't find a matching dish in the menu."
    lines = ["I couldn't get a response from Gemini, so here are the matching menu records:"]
    for hit in hits:
        lines.append(f"\n**{hit.get('name')} ({hit.get('dish_id')})** — ₹{hit['price']:g}" if hit.get("price") is not None else f"\n**{hit.get('name')} ({hit.get('dish_id')})**")
        if hit.get("description"):
            lines.append(hit["description"])
        for key, label in (("cuisine", "Cuisine"), ("category", "Category")):
            if hit.get(key):
                lines.append(f"{label}: {hit[key]}")
        dietary = [f"{label}: {hit[key]}" for key, label in (
            ("vegetarian", "vegetarian"), ("vegan", "vegan"), ("jain", "Jain"),
            ("gluten_free", "gluten-free"), ("dairy_free", "dairy-free"), ("egg_free", "egg-free"),
        ) if hit.get(key) is not None]
        if dietary:
            lines.append("Menu dietary labels: " + ", ".join(dietary))
        conditions = hit.get("dietary_conditions")
        if conditions:
            lines.append(f"Dietary conditions: {conditions}")
        if hit.get("allergens") is not None:
            lines.append(f"Confirmed allergens: {hit['allergens']}")
        if hit.get("may_contain_allergens") is not None:
            lines.append(f"May contain: {hit['may_contain_allergens']}")
    return "\n".join(lines)


def answer_question(question: str, history: list[dict] | None = None) -> str:
    """Retrieve grounded menu evidence with one common search flow, then ask Gemini to answer."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question about the menu."
    allergen_question = bool(re.search(r"allergen|allergy|safe to eat|allergy-safe|free from", question, re.I))
    def finish(text: str) -> str:
        return _allergen_warning(text) if allergen_question else text
    history = history or []
    context = _history_text(history)
    filters = _filters(question)
    requested_ids = list(dict.fromkeys(
        f"D{int(number):03d}" for number in re.findall(r"\bD\s*0*(\d{1,3})\b", question, re.I)
    ))
    if requested_ids:
        hits = []
        for dish_id in requested_ids:
            hits.extend(exact_lookup(dish_id))
        found_ids = {hit.get("dish_id") for hit in hits}
        missing_ids = [dish_id for dish_id in requested_ids if dish_id not in found_ids]
        if missing_ids:
            return finish("I couldn't find dish ID " + ", ".join(missing_ids) + " in the current menu index.")
    else:
        hits = search_menu(question, top_k=10, filters=filters)

    if not hits:
        return finish("I couldn't find a menu record that satisfies those terms and stated filters. Unknown dietary values do not pass a positive dietary filter.")

    # Keep citations out of model input; the app adds verified citations beside selected dishes.
    evidence = _evidence_text(hits)
    system = (
        "You are MenuLens, a warm, concise conversational menu assistant. Answer in natural language rather than copying or listing raw database records. "
        "Use only facts supported by the supplied menu records. Never invent dishes, ingredients, prices, descriptions, or dietary properties. "
        "Only name dishes that appear in the supplied records. If the user asks for an exact dish that is not present, say that no exact match was retrieved and clearly label any named alternatives as similar suggestions. "
        "Include the dish ID when naming a dish. "
        "For recommendation or suggestion requests, offer 3 to 5 distinct dishes when that many relevant records are available; offer fewer when fewer records meet the request. "
        "For each recommendation, briefly explain why it matches the user's request using specific details in its menu description, and include its price and description when available. "
        "Do not label a dish spicy, warming, light, creamy, or similar unless the menu description supports that wording. Keep menu facts distinct from any inference; explicitly qualify an inference and explain what it is based on. "
        "When comparing dishes, compare only the requested dishes present in the records and state when details are missing. Explain dietary preparation conditions and never present unknown suitability as confirmed. "
        "Use the original question and conversation to understand follow-ups, but answer only from the retrieved evidence. End with one useful follow-up question when it would help the diner choose. "
        "Do not add source lines, PDF citations, or a source section; the app adds a PDF citation beside each dish it identifies in your answer."
    )
    if allergen_question:
        disclaimer = get_menu_manifest().get("menu_disclaimer") or MENU_ALLERGEN_DISCLAIMER
        evidence += "\n\nMenu-level allergen disclaimer: " + disclaimer
        system += " Allergen notes are indicative; never call a dish allergen-free based on missing notes. Do not repeat the menu-wide allergen reminder; the app will append it."
    try:
        answer = _gemini_answer(question, context, evidence, system)
    except Exception as exc:
        if "api key" in str(exc).lower() or "api_key" in str(exc).lower():
            answer = _fallback(hits, "missing API key")
        else:
            answer = _fallback(hits, str(exc))
    return finish(_cite_selected_dishes(_remove_generated_sources(answer), hits))
