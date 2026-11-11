"""Convert Gradio's session history to LangChain messages safely."""

from langchain_core.messages import AIMessage, HumanMessage


def _plain_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def to_langchain_messages(history: list[dict], limit: int = 12):
    """Keep the latest text-only turns; ignore non-text UI payloads."""
    messages = []
    for item in history[-limit:]:
        text = _plain_text(item.get("content", "")).strip()
        if not text:
            continue
        if item.get("role") == "user":
            messages.append(HumanMessage(content=text))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=text))
    return messages
