"""Gradio chat interface for MenuLens."""

import gradio as gr

from config import GEMINI_FALLBACK_MODEL, GEMINI_MODEL, GRADIO_SHARE, PDF_PATH
from retrieval.query_processor import answer_question
from utils.citations import linkify_page_citations, menu_page_url
from utils.helpers import index_ready


def respond(message: str, history: list[dict]) -> str:
    if not (message or "").strip():
        return "Please enter a question about the menu."
    if not index_ready():
        return "The menu index isn't ready yet. Build it with `python -m indexing.build_index`, then restart this app."
    try:
        return linkify_page_citations(answer_question(message, history or []))
    except Exception as exc:
        error_text = str(exc).lower()
        if "429" in error_text or "resource_exhausted" in error_text:
            return "Gemini is temporarily rate-limited. Please try again shortly."
        if any(marker in error_text for marker in ("503", "unavailable", "504", "deadline_exceeded", "timed out", "timeout")):
            return "Gemini is temporarily unavailable. Please try again shortly."
        if "api key" in error_text or "api_key" in error_text:
            return "Gemini isn't configured yet. Add your Google API key to the local `.env` file and restart MenuLens."
        if any(marker in error_text for marker in ("huggingface", "sentence-transformers", "model files", "local_files_only")):
            return "The local semantic-search model isn't available yet. Build the index while connected to download it, then restart MenuLens. Exact dish lookups remain available from the saved index."
        # Do not return provider exceptions, which can contain request or account details.
        return "I couldn't complete that menu search. Please try again shortly."


def create_app() -> gr.Blocks:
    """Build the UI on demand so importing query helpers does not initialize Gradio."""
    with gr.Blocks(title="MenuLens") as demo:
        gr.Markdown(
            "# MenuLens\nAsk about dishes in the prepared menu. Answers use the menu records and cite the PDF page. "
            "Search results include dish IDs and PDF pages. Price and stated dietary filters are available."
        )
        with gr.Accordion("View the menu PDF here", open=False, elem_id="menu-pdf-accordion"):
            gr.HTML(
                f'<iframe id="menulens-pdf-viewer" name="menulens-pdf-viewer" title="Menu PDF" '
                f'src="{menu_page_url(1)}" style="width:100%;height:720px;border:0;"></iframe>'
            )
        gr.ChatInterface(
            fn=respond,
            title="Menu assistant",
            examples=["What dishes sound spicy?", "Tell me about the pasta options.", "Compare Paneer Butter Masala and Dal Makhani."],
            save_history=False,
        )
    return demo


if __name__ == "__main__":
    demo = create_app()
    print(f"Gemini model: {GEMINI_MODEL} (fallback: {GEMINI_FALLBACK_MODEL})")
    if index_ready():
        print("Menu index is ready. The embedding model will load only if a semantic search needs it.")
    else:
        print("Menu index not found. Build it with: python -m indexing.build_index")
    print(f"Gradio public sharing enabled: {GRADIO_SHARE}")
    demo.launch(share=GRADIO_SHARE, allowed_paths=[str(PDF_PATH.resolve())])
