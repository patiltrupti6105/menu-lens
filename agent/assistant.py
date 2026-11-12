"""Fast direct RAG for simple questions; bounded tool calling for complex ones."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools import MENU_TOOLS
from config import (
    GEMINI_FALLBACK_MODEL,
    GEMINI_MODEL,
    GEMINI_REQUEST_TIMEOUT,
    GEMINI_THINKING_LEVEL,
    MAX_AGENT_TURNS,
)
from memory.chat_history import to_langchain_messages
from retrieval.search import format_hit, search_menu
from utils.citations import linkify_page_citations, menu_page_citation

SYSTEM_PROMPT = """You are MenuLens, an assistant for the menu PDF indexed by this app.
Use the menu tools for factual menu questions. Treat tool results as the only source
for dishes, descriptions, prices, ingredients, dietary labels, and allergens. Never
guess missing details. State clearly when a requested fact is not present. Cite each
dish using the exact PDF page link included in tool results, preserving its link markup.
Pass requested price, cuisine, or dietary constraints
to search_dishes. Do not claim a dish meets a filter unless the verified menu fields support it.
For comparisons, compare only facts found for each requested dish. Be concise and
retain conversation context for follow-up questions."""

DIRECT_PROMPT = """You are MenuLens, a menu assistant. Answer using only the retrieved menu
records included in the user's latest message. Do not invent dishes, prices, ingredients,
dietary labels, or allergens. If the records do not support an answer, say what is missing.
Cite each dish using the PDF page link in its retrieved record. Preserve its link markup.
Be concise."""

_COMPARISON = re.compile(
    r"\b(compare|comparison|cheaper|cheapest|lowest[- ]priced|most expensive|"
    r"which costs? less|which costs? more|difference between)\b",
    re.IGNORECASE,
)
_STRUCTURED_FILTER = re.compile(
    r"\b(vegetarian|vegan|jain|gluten[- ]free|dairy[- ]free|under|below|"
    r"less than|at most|no more than|between|over|above|more than|within budget|"
    r"price range|maximum price|minimum price)\b|₹\s*\d+|\b\d+\s*(?:rupees|rs\.?|inr)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_FOLLOWUP = re.compile(
    r"\b(which one|what about (?:it|that|them|those)|the other one|same one|"
    r"that dish|those dishes|and that|how about it)\b",
    re.IGNORECASE,
)


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def _get_llm(model_name: str = GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        thinking_config={"thinking_level": GEMINI_THINKING_LEVEL},
        max_output_tokens=512,
        timeout=GEMINI_REQUEST_TIMEOUT,
        max_retries=1,
    )


def _invoke_with_fallback(messages, model_name: str, use_tools: bool = False):
    """Try the configured Flash model, switching once after a temporary error."""
    candidates = list(dict.fromkeys((model_name, GEMINI_FALLBACK_MODEL)))
    for index, candidate in enumerate(candidates):
        model = _get_llm(candidate)
        if use_tools:
            model = model.bind_tools(MENU_TOOLS)
        try:
            return model.invoke(messages), candidate
        except Exception as exc:
            if index + 1 >= len(candidates) or not _is_temporary_gemini_error(str(exc).lower()):
                raise
            print(
                f"Gemini model {candidate} is temporarily unavailable; "
                f"trying {candidates[index + 1]}.",
                flush=True,
            )
    raise RuntimeError("No Gemini model was available.")


def _needs_agent(question: str, history: list[dict]) -> bool:
    """Use tools when constraints or prior results require an explicit workflow."""
    if _COMPARISON.search(question) or _STRUCTURED_FILTER.search(question):
        return True
    if history and (
        _CONTEXTUAL_FOLLOWUP.search(question)
        or len(question.split()) <= 4
    ):
        return True
    return False


def _answer_directly(question: str, history: list[dict]) -> str:
    """Retrieve first, then use one Gemini call to phrase the grounded answer."""
    hits = search_menu(question, top_k=3)
    if not hits:
        return "I couldn't find supporting text in the menu for that question."
    evidence = "\n\n---\n\n".join(format_hit(hit) for hit in hits)
    messages = [SystemMessage(content=DIRECT_PROMPT)]
    messages.extend(to_langchain_messages(history))
    messages.append(
        HumanMessage(
            content=(
                f"Question: {question}\n\nRetrieved menu records (use only these as evidence):\n"
                f"{evidence}"
            )
        )
    )
    try:
        response, _ = _invoke_with_fallback(messages, GEMINI_MODEL)
        answer = _text_content(response.content)
        return linkify_page_citations(answer, hits) if answer else _direct_fallback(hits, "empty response")
    except Exception as exc:
        error_text = str(exc).lower()
        if _is_temporary_gemini_error(error_text):
            return _direct_fallback(hits, error_text)
        raise


def _is_temporary_gemini_error(error_text: str) -> bool:
    """Recognize common temporary overload, rate-limit, and timeout responses."""
    return any(
        marker in error_text
        for marker in (
            "429", "resource_exhausted", "500", "502", "503", "504",
            "unavailable", "deadline_exceeded", "timed out", "timeout",
        )
    )


def _direct_fallback(hits: list[dict], reason: str = "") -> str:
    """Show retrieval-only suggestions if Gemini cannot phrase an answer."""
    reason_lower = reason.lower()
    if "503" in reason_lower or "unavailable" in reason_lower:
        cause = "Gemini is temporarily busy."
    elif "429" in reason_lower or "resource_exhausted" in reason_lower:
        cause = "Gemini is temporarily rate-limited."
    elif "empty response" in reason_lower:
        cause = "Gemini returned an empty response."
    else:
        cause = "Gemini took too long to respond."
    lines = [
        f"{cause} These are the closest matches from the local menu search, "
        "not a verified answer to every part of your request:"
    ]
    for hit in hits:
        name = hit.get("name") or hit.get("dish_id") or "Menu item"
        price = hit.get("price")
        price_text = f"₹{price:g}" if isinstance(price, (int, float)) else "price not listed"
        description = next(
            (line.removeprefix("Description: ") for line in hit["text"].splitlines()
             if line.startswith("Description: ")),
            "",
        )
        page = hit.get("page_number")
        detail = f" — {description}" if description else ""
        source = menu_page_citation(page, name)
        lines.append(f"- **{name}** ({price_text}){detail}. Source: {source}.")
    return "\n".join(lines)


def _answer_with_agent(question: str, history: list[dict]) -> str:
    """Answer with a bounded number of read-only tool-calling turns."""
    active_model = GEMINI_MODEL
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(to_langchain_messages(history))
    messages.append(HumanMessage(content=question))

    for _ in range(MAX_AGENT_TURNS):
        response, active_model = _invoke_with_fallback(
            messages, active_model, use_tools=True
        )
        messages.append(response)
        calls = getattr(response, "tool_calls", []) or []
        if not calls:
            return _text_content(response.content) or "I couldn't form a menu answer from the available information."

        for call in calls:
            tool = next((candidate for candidate in MENU_TOOLS if candidate.name == call.get("name")), None)
            if tool is None:
                output = "That tool is not available."
            else:
                try:
                    output = str(tool.invoke(call.get("args", {})))
                except Exception as exc:
                    output = f"Menu search could not run: {exc}"
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    # One final non-tool response after the tool iteration budget is reached.
    final, _ = _invoke_with_fallback(
        messages + [SystemMessage(content="Give the best grounded answer now. Do not call more tools.")],
        active_model,
    )
    return _text_content(final.content) or "I couldn't complete that menu search."


def answer_question(question: str, history: list[dict] | None = None) -> str:
    """Use one Gemini call for simple retrieval; use the agent when tools add value."""
    history = history or []
    if _needs_agent(question, history):
        return _answer_with_agent(question, history)
    return _answer_directly(question, history)
