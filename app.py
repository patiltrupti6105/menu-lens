"""Gradio chat interface for MenuLens."""

import gradio as gr

from config import GEMINI_FALLBACK_MODEL, GEMINI_MODEL
from utils.helpers import index_ready


def respond(message: str, history: list[dict]) -> str:
    if not index_ready():
        return "The menu index isn't ready yet. Build it with `python -m indexing.build_index`, then restart this app."
    try:
        # Defer the LangChain/Gemini imports until the first chat message so UI startup stays quick.
        from agent.assistant import answer_question

        return answer_question(message, history)
    except Exception as exc:
        message = str(exc)
        error_text = message.lower()
        if "503" in error_text or "unavailable" in error_text:
            return (
                "Gemini is temporarily busy (503). Please try again shortly. "
                "This request needs Gemini to apply the requested filters or comparison, "
                "so I haven't substituted approximate matches."
            )
        if "429" in error_text or "resource_exhausted" in error_text:
            return "Gemini is temporarily rate-limited. Please try again shortly."
        if any(marker in error_text for marker in ("504", "deadline_exceeded", "timed out", "timeout")):
            return "Gemini took too long to respond. Please try again shortly."
        if "API key" in message or "api_key" in message.lower():
            return "Gemini isn't configured yet. Add your Google API key to the local `.env` file and restart MenuLens."
        return f"I couldn't complete that request: {message}"


with gr.Blocks(title="MenuLens") as demo:
    gr.Markdown(
        "# MenuLens\nAsk about dishes in the prepared menu. Answers use the PDF text and cite its page when available. "
        "Search results include dish IDs and PDF pages. Price and stated dietary filters are available."
    )
    gr.ChatInterface(
        fn=respond,
        title="Menu assistant",
        examples=[
            "What dishes sound spicy?",
            "Tell me about the pasta options.",
            "Compare the two vegetarian dishes.",
        ],
        save_history=False,
    )


if __name__ == "__main__":
    print(f"Gemini model: {GEMINI_MODEL} (fallback: {GEMINI_FALLBACK_MODEL})")
    if index_ready():
        from retrieval.vector_store import warm_up

        print("Loading the local menu search model...")
        warm_up()
        print("Menu search is ready.")
    else:
        print("Menu index not found. Build it with: python -m indexing.build_index")
    demo.launch(share=True)
