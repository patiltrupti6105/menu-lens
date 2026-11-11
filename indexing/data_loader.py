"""Extract the PDF's explicit dish entries into validated dish records."""

from pathlib import Path
import re

from pypdf import PdfReader
from pydantic import ValidationError

from models.dish_schema import Dish


DISH_ID = re.compile(r"^D(\d{3})\s*$")
PRICE = re.compile(r"₹\s*([\d,]+(?:\.\d{1,2})?)")
CUISINE_HEADINGS = {
    "SOUPS & SALADS": None,
    "INDIAN CUISINE": "Indian",
    "CHINESE CUISINE": "Chinese",
    "ITALIAN CUISINE": "Italian",
    "CONTINENTAL & AMERICAN CUISINE": "Continental & American",
    "DESSERTS & BAKES": None,
    "THAI CUISINE": "Thai",
    "MEXICAN CUISINE": "Mexican",
    "BEVERAGES": None,
}
RESTAURANT_NAME = "The MenuLens Kitchen"


def _clean_description(lines: list[str]) -> str:
    kept = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Allergen note:") or PRICE.search(line):
            continue
        kept.append(line)
    return " ".join(kept)


def _allergens(entry: str) -> list[str] | None:
    match = re.search(r"Allergen note:\s*(.*?)(?:\n|$)", entry, flags=re.IGNORECASE)
    if not match:
        return None
    note = match.group(1).strip().rstrip(".")
    note = re.sub(r"^contains\s+", "", note, flags=re.IGNORECASE)
    return [part.strip() for part in re.split(r",|\s+and\s+", note) if part.strip()] or None


def _diet_flags(entry: str) -> tuple[bool | None, bool | None]:
    normalized = " ".join(entry.lower().split())
    if "non-vegetarian" in normalized or "non vegetarian" in normalized:
        vegetarian = False
    elif re.search(r"\bvegetarian\b", normalized):
        vegetarian = True
    else:
        vegetarian = None

    if re.search(r"\bvegan\s+when\b", normalized):
        vegan = None
    elif re.search(r"\bvegan\b", normalized):
        vegan = True
    else:
        vegan = None
    if vegan is True and vegetarian is None:
        vegetarian = True
    return vegetarian, vegan


def load_dishes(pdf_path: Path) -> list[Dish]:
    """Parse menu entries beginning with D### and validate them with Pydantic."""
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Menu PDF not found: {pdf_path}. Set MENULENS_PDF_PATH to its location."
        )

    reader = PdfReader(str(pdf_path))
    dishes: list[Dish] = []
    errors: list[str] = []
    current_cuisine: str | None = None
    current_id: str | None = None
    current_page = 0
    entry_lines: list[str] = []

    def save_entry() -> None:
        if current_id is None:
            return
        entry = "\n".join(entry_lines)
        lines = [line.strip() for line in entry_lines if line.strip()]
        if not lines:
            errors.append(f"{current_id} on page {current_page}: dish name missing")
            return
        name = lines[0]
        price_match = PRICE.search(entry)
        price = float(price_match.group(1).replace(",", "")) if price_match else None
        vegetarian, vegan = _diet_flags(entry)
        try:
            dishes.append(
                Dish(
                    dish_id=current_id,
                    restaurant_name=RESTAURANT_NAME,
                    name=name,
                    description=_clean_description(lines[1:]) or None,
                    price=price,
                    cuisine=current_cuisine,
                    ingredients=[],
                    vegetarian=vegetarian,
                    vegan=vegan,
                    allergens=_allergens(entry),
                    dietary_source="Menu PDF",
                    source_reference=f"{pdf_path.name}, page {current_page}",
                    page_number=current_page,
                )
            )
        except ValidationError as exc:
            errors.append(f"{current_id} on page {current_page}: {exc}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped in CUISINE_HEADINGS:
                save_entry()
                current_id = None
                entry_lines = []
                current_cuisine = CUISINE_HEADINGS[stripped]
                continue
            id_match = DISH_ID.match(stripped)
            if id_match:
                save_entry()
                current_id = f"D{id_match.group(1)}"
                current_page = page_number
                entry_lines = []
            elif current_id is not None:
                entry_lines.append(stripped)
        save_entry()
        current_id = None
        entry_lines = []

    if errors:
        raise ValueError("Could not validate all menu entries:\n- " + "\n- ".join(errors))
    if not dishes:
        raise ValueError("No dish IDs like D001 were found in the PDF text. The PDF may need OCR.")
    ids = [dish.dish_id for dish in dishes]
    if len(ids) != len(set(ids)):
        raise ValueError("The PDF contains duplicate dish IDs; fix the menu before indexing.")
    return dishes
