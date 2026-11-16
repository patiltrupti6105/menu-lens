"""Predictable query routing, grounded retrieval, and direct Gemini response generation."""

import re
from typing import Any

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODEL, GEMINI_MODEL, GEMINI_REQUEST_TIMEOUT, GEMINI_THINKING_LEVEL
from retrieval.search import exact_lookup, filter_dishes, list_dishes, normalize_name, resolve_dish_name, search_menu
from utils.citations import menu_page_citation
from indexing.data_loader import MENU_ALLERGEN_DISCLAIMER
from retrieval.vector_store import get_menu_manifest


_COMPARISON_REQUEST = re.compile(
    r"\b(compare|comparison|difference between|which costs?|cheaper|more expensive)\b", re.I
)
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


def _comparison_pair(question: str) -> list[str] | None:
    """Extract two explicitly named items without using semantic substitution."""
    patterns = (
        r"\bcompare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
        r"\bcompare\s+(.+?)\s+with\s+(.+?)(?:[?.!]|$)",
        r"\bdifference between\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
    )
    match = next((match for pattern in patterns if (match := re.search(pattern, question, re.I))), None)
    if not match:
        return None
    return [
        re.sub(r"^(?:the|a|an)\s+", "", re.sub(r",?\s*(?:please|for me|thanks)$", "", part.strip(), flags=re.I), flags=re.I)
        for part in match.groups()
    ]


def _compare_all_category(question: str) -> str | None:
    for alias, category in sorted(CATEGORY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b(?:all|the)\s+{re.escape(alias)}\b", question, re.I):
            return category
    return None


def _mentioned_dishes(question: str, context: str) -> tuple[list[dict], list[str]]:
    recent_context = "\n".join(context.splitlines()[-2:])
    combined = f"{question}\n{recent_context}"
    hits = list_dishes()
    mentioned = [hit for hit in hits if hit.get("name") and hit["name"].casefold() in combined.casefold()]
    mentioned.sort(key=lambda hit: len(hit["name"]), reverse=True)
    # Longest name first and avoid treating a name contained within another name as a second request.
    chosen, occupied = [], []
    for hit in mentioned:
        name = hit["name"].casefold()
        if not any(name in earlier or earlier in name for earlier in occupied):
            chosen.append(hit); occupied.append(name)
    ids = re.findall(r"\bD\s*0*(\d{1,3})\b", question, re.I)
    for number in ids:
        target = f"D{int(number):03d}"
        match = next((hit for hit in hits if hit["dish_id"] == target), None)
        if match and match not in chosen:
            chosen.append(match)
    # Exact names mentioned in the current user request take precedence over prior context.
    current = [hit for hit in chosen if hit["name"].casefold() in question.casefold() or hit["dish_id"] in question]
    return (current or chosen), []


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
    """Route known operations with Python, then ask Gemini to phrase grounded evidence."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question about the menu."
    allergen_question = bool(re.search(r"allergen|allergy|safe to eat|allergy-safe|free from", question, re.I))
    def finish(text: str) -> str:
        return _allergen_warning(text) if allergen_question else text
    history = history or []
    context = _history_text(history)
    filters = _filters(question)
    explicit_lookup = re.search(r"\b(?:what is|tell me about|find|details for|details about)\s+(.+?)(?:[?.!]|$)", question, re.I)
    mentioned, _ = _mentioned_dishes(question, context)
    current_names = [hit for hit in mentioned if hit.get("name", "").casefold() in question.casefold() or hit.get("dish_id") in question]
    contextual_followup = bool(
        mentioned and not current_names and not explicit_lookup and
        (len(question.split()) <= 3 or re.search(r"\b(it|this|that|them|those|same one|which one|what about|how about|and that|and it)\b", question, re.I))
    )
    standalone_id = re.fullmatch(r"\s*(?:(?:tell me about|what is|find|details for|details about)\s+)?D\s*0*(\d{1,3})\s*[?.!]*\s*", question, re.I)
    comparison = bool(_COMPARISON_REQUEST.search(question))
    exact_name_question = any(
        normalize_name(question) == normalize_name(hit.get("name", ""))
        for hit in list_dishes()
    )
    exact = []
    if not comparison and (explicit_lookup or standalone_id or exact_name_question):
        exact = exact_lookup(question)
        if not exact and explicit_lookup and not re.search(r"\bD\s*0*\d{1,3}\b", explicit_lookup.group(1), re.I):
            exact = resolve_dish_name(explicit_lookup.group(1))

    explicit_id_lookup = bool(standalone_id or (
        explicit_lookup and re.search(r"\bD\s*0*\d{1,3}\b", explicit_lookup.group(1), re.I)
    ))
    if comparison:
        requested_names = _comparison_pair(question)
        if requested_names:
            hits = []
            missing = []
            for requested in requested_names:
                found = resolve_dish_name(requested)
                if found:
                    hits.extend(found)
                else:
                    missing.append(requested)
            if missing:
                return finish("I couldn't uniquely match these requested dishes: " + ", ".join(missing) + ". Please use a full menu name or dish ID; I haven't substituted similar dishes.")
            hits = list({hit["dish_id"]: hit for hit in hits}.values())
        else:
            group_category = _compare_all_category(question)
            if group_category:
                hits = filter_dishes({**filters, "category": group_category})
            elif mentioned and len(mentioned) >= 2:
                hits = []
                for requested in mentioned:
                    hits.extend(exact_lookup(requested["dish_id"]))
            elif filters:
                hits = search_menu(question, top_k=5, filters=filters)
            else:
                hits = search_menu(question, top_k=5)
        if len(hits) < 2:
            return finish("I couldn't find enough matching menu dishes to compare. Try naming two dishes or a menu category.")
        intent = "Compare only the retrieved menu records. Explain differences using menu-supported descriptions, prices, cuisine, dietary labels, and allergens when available. Do not infer missing facts."
    elif exact:
        hits = exact
        intent = "Answer about this exact requested dish using its menu record. Do not replace it."
    elif explicit_id_lookup:
        return finish("I couldn't find that dish ID in the current menu index.")
    elif contextual_followup:
        hits = []
        for requested in mentioned:
            hits.extend(exact_lookup(requested["dish_id"]))
        intent = "Answer this follow-up about the dish or dishes in the immediately preceding conversation using their exact menu records."
    else:
        intent = "Answer the user's menu question using only the retrieved records."
        if filters:
            hits = search_menu(question, top_k=8, filters=filters)
        else:
            hits = search_menu(f"{context[-800:]}\n{question}" if context else question, top_k=5)
        if not hits:
            # High-confidence typo / normalized exact name can still succeed after semantic retrieval.
            hits = exact_lookup(question)
        if not hits:
            return finish("I couldn't find a menu record that satisfies those terms and stated filters. Unknown dietary values do not pass a positive dietary filter.")
        if explicit_lookup:
            intent = "The exact requested name was not found. Clearly label any retrieved records as similar suggestions, never as the requested dish."

    # Keep citations out of model input; the app adds verified citations beside selected dishes.
    evidence = "\n\n---\n\n".join(hit.get("text", "") for hit in hits)
    system = (
        "You are MenuLens, a warm, concise conversational menu assistant. Answer in natural language rather than copying or listing raw database records. "
        "Use only facts supported by the supplied menu records. Never invent dishes, ingredients, prices, descriptions, or dietary properties. "
        "For recommendation or suggestion requests, offer 3 to 5 distinct dishes when that many relevant records are available; offer fewer when fewer records meet the request. "
        "For each recommendation, briefly explain why it matches the user's request using specific details in its menu description, and include its price and description when available. "
        "Do not label a dish spicy, warming, light, creamy, or similar unless the menu description supports that wording. Keep menu facts distinct from any inference; explicitly qualify an inference and explain what it is based on. "
        "For exact dish lookups, answer the specific request without adding unrelated recommendations. Mention dish IDs. Explain dietary preparation conditions and never present unknown suitability as confirmed. "
        "For comparisons, use only the actual records supplied and state when details are missing. End with one useful follow-up question when it would help the diner choose. "
        "Do not add source lines, PDF citations, or a source section; the app adds a PDF citation beside each dish it identifies in your answer."
    )
    system += " Specific task: " + intent
    if explicit_lookup and not exact:
        system += " Start by saying the exact requested dish name was not found. Any records shown are only possible similar matches."
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
        if explicit_lookup and not exact:
            requested = explicit_lookup.group(1).strip()
            answer = f"I couldn't find an exact name match for **{requested}**. These are only similar menu records, not confirmation that the requested dish exists.\n\n" + answer
    return finish(_cite_selected_dishes(_remove_generated_sources(answer), hits))
