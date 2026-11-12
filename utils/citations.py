"""Links from answer citations to the source menu PDF inside the Gradio app."""

import html
import re
from urllib.parse import quote

from config import PDF_PATH


def menu_pdf_file_url() -> str:
    """Return the app's Gradio file route for the single source PDF."""
    return f"/gradio_api/file={quote(str(PDF_PATH.resolve()), safe='/')}"


def menu_page_url(page_number: int, dish_name: str | None = None) -> str:
    """Return the file route with native PDF page and optional text-highlight fragments."""
    fragment = f"#page={page_number}"
    if dish_name:
        fragment += f":~:text={quote(dish_name, safe='')}"
    return menu_pdf_file_url() + fragment


def menu_page_citation(page_number: int | None, dish_name: str | None = None) -> str:
    if not page_number:
        return "Menu PDF"
    label = f"Menu PDF, page {page_number}"
    if dish_name:
        label += f" — {dish_name}"
    target = menu_page_url(page_number, dish_name)
    return (
        f'<a href="{html.escape(target, quote=True)}" '
        f'target="menulens-pdf-viewer">{html.escape(label)}</a>'
    )


_LINK_OR_PAGE_REFERENCE = re.compile(
    r"(?P<link><a\b[^>]*>.*?</a>|\[[^\]]+\]\([^)]+\))|"
    r"(?P<reference>(?:(?:menu(?:lens_menu)?\.pdf|menu\s+pdf)\s*,?\s*)?"
    r"page\s*#?\s*(?P<page>\d+))",
    re.IGNORECASE | re.DOTALL,
)


def linkify_page_citations(answer: str, hits: list[dict] | None = None) -> str:
    """Link plain page references, attaching the nearby retrieved dish when known."""
    hits = hits or []

    def replace(match: re.Match) -> str:
        if match.group("link"):
            return match.group("link")

        page = int(match.group("page"))
        nearby_text = answer[max(0, match.start() - 240):match.start()].casefold()
        page_hits = [hit for hit in hits if hit.get("page_number") == page]
        named_hits = [
            hit for hit in page_hits
            if hit.get("name") and hit["name"].casefold() in nearby_text
        ]
        selected = named_hits[-1] if named_hits else page_hits[0] if len(page_hits) == 1 else None
        dish_name = selected.get("name") if selected else None
        return menu_page_citation(page, dish_name)

    return _LINK_OR_PAGE_REFERENCE.sub(replace, answer)
