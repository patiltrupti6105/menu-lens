"""Parse the MenuLens menu PDF into explicit, validated dish records."""

from pathlib import Path
import re

from pypdf import PdfReader
from pydantic import ValidationError

from models.dish_schema import Dish

DISH_ID = re.compile(r"^D(\d{3})\s*$")
PRICE = re.compile(r"₹\s*([\d,]+(?:\.\d{1,2})?)")
RESTAURANT_NAME = "The MenuLens Kitchen"

# The PDF's section names are data, not inferred dish cuisines.
CUISINES = {
    "INDIAN CUISINE": "Indian", "CHINESE CUISINE": "Chinese",
    "ITALIAN CUISINE": "Italian", "CONTINENTAL & AMERICAN CUISINE": "Continental & American",
    "THAI CUISINE": "Thai", "MEXICAN CUISINE": "Mexican",
}
NON_CUISINE_HEADINGS = {"SOUPS & SALADS", "DESSERTS & BAKES", "BEVERAGES"}
CATEGORIES = {
    "Soups", "Salads", "Starters & Tandoor", "Main Course - Vegetarian",
    "Main Course - Non-Vegetarian", "Biryani & Rice", "Breads from the Tandoor",
    "Mithai & Indian Desserts", "Indo-Chinese Starters", "Noodles, Rice & Wok Mains",
    "Antipasti", "Wood-Fired Pizza", "Pasta & Baked Classics", "Dolci",
    "Grills, Burgers & Comfort Plates", "Cakes, Brownies & Sweet Finales",
    "Curries, Noodles & Street Plates", "Tacos, Burritos & Antojitos",
    "Coffee & Hot Drinks", "Teas & Iced Teas", "Mocktails & Coolers", "Shakes & Lassi",
}
MENU_ALLERGEN_DISCLAIMER = (
    "Allergen notes are indicative and based on standard recipes. Our kitchen handles gluten, milk, egg, soy, "
    "peanuts, tree nuts, sesame, fish, shellfish and mustard, so cross-contact is possible in any dish. "
    "Where an allergen note is not shown for a dish, please confirm with our team before ordering."
)


def load_menu_disclaimer(pdf_path: Path) -> str | None:
    """Return the menu-level allergen disclaimer when it is present in the PDF."""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    if "cross-contact is possible in any dish" in " ".join(text.lower().split()):
        return MENU_ALLERGEN_DISCLAIMER
    return None


def _allergen_lists(entry: str) -> tuple[list[str] | None, list[str] | None]:
    match = re.search(r"Allergen note:\s*(.*?)(?:\n|$)", entry, re.IGNORECASE)
    if not match:
        return None, None
    note = match.group(1).strip().rstrip(".")
    confirmed_match = re.search(r"contains\s+(.+?)(?=\s+and\s+may contain\s+|$)", note, re.I)
    possible_match = re.search(r"may contain\s+(.+)$", note, re.I)

    def split_values(value: str | None) -> list[str] | None:
        if not value:
            return None
        values = re.split(r",\s*|\s+and\s+", value.strip().rstrip("."), flags=re.I)
        return [v.strip().lower() for v in values if v.strip()] or None

    if confirmed_match:
        confirmed = split_values(confirmed_match.group(1))
    else:
        confirmed = split_values(re.sub(r"^contains\s+", "", note, flags=re.I))
    return confirmed, split_values(possible_match.group(1) if possible_match else None)


def _dietary_data(text: str) -> dict:
    lower = " ".join(text.lower().split())
    conditions: list[str] = []
    # Preserve each conditional claim verbatim enough to explain required preparation.
    conditional_patterns = (
        r"\bvegan(?:\s+and\s+gluten[- ]free)?\s+(?:when|if|unless|only when|only if)[^.]*",
        r"\bjain-friendly\b[^.]*?\b(?:when|if|unless|without|on request)[^.]*",
        r"\b(?:gluten[- ]free|dairy[- ]free|egg[- ]free)\s+(?:when|if|unless)[^.]*",
    )
    for pattern in conditional_patterns:
        for match in re.finditer(pattern, text, re.I):
            clause = " ".join(match.group(0).split()).rstrip("., ")
            if not any(clause.casefold() in other.casefold() or other.casefold() in clause.casefold() for other in conditions):
                conditions.append(clause)
    if re.search(r"\bvegan[- ]friendly\b", lower):
        conditions.append("Described as vegan-friendly; unconditional vegan suitability is not confirmed")

    vegetarian = False if re.search(r"\bnon[- ]vegetarian\b", lower) else (
        True if re.search(r"\bvegetarian\b", lower) else None
    )
    vegan = None
    if re.search(r"\bvegan(?:\s+and\s+gluten[- ]free)?\s+(?:when|if|unless|only when|only if)\b", lower):
        vegan = None
    elif re.search(r"\bvegan[- ]friendly\b", lower):
        vegan = None
    elif re.search(r"\bvegan\b", lower):
        vegan = True
    elif vegetarian is False:
        vegan = False
    else:
        confirmed, _ = _allergen_lists(text)
        if confirmed and any(value in confirmed for value in ("milk", "egg", "fish", "shellfish")):
            vegan = False
    if vegan is True and vegetarian is None:
        vegetarian = True

    jain = None
    if re.search(r"\b(?:jain-friendly|suitable for jain diners|jain diners)\b", lower):
        if not re.search(r"\bjain-friendly\b[^.]*\b(?:when|if|unless|on request|without)\b", lower):
            jain = True

    def explicit_flag(label: str) -> bool | None:
        if re.search(rf"\b{label}[- ]free\s+(?:if|when|unless)\b", lower):
            return None
        if re.search(rf"\b{label}[- ]free\b", lower):
            return True
        # An explicit allergen note can establish that this menu version contains it.
        allergens, possible = _allergen_lists(text)
        allergen = {"gluten": "gluten", "dairy": "milk", "egg": "egg"}[label]
        if allergens and allergen in allergens:
            return False
        return None

    return {
        "vegetarian": vegetarian, "vegan": vegan, "jain": jain,
        "gluten_free": explicit_flag("gluten"),
        "dairy_free": explicit_flag("dairy"), "egg_free": explicit_flag("egg"),
        "dietary_conditions": conditions,
    }


def _clean_description(lines: list[str]) -> str | None:
    kept = []
    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("allergen note:") or PRICE.search(line):
            continue
        kept.append(line)
    return " ".join(kept) or None


def load_dishes(pdf_path: Path) -> list[Dish]:
    """Read menu sections and D### records; fail if any dish record is malformed."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Menu PDF not found: {pdf_path}. Set MENULENS_PDF_PATH to its location.")
    reader = PdfReader(str(pdf_path))
    dishes: list[Dish] = []
    errors: list[str] = []
    cuisine: str | None = None
    category: str | None = None
    current_id: str | None = None
    current_page = 0
    entry_lines: list[str] = []

    def save_entry() -> None:
        if current_id is None:
            return
        lines = [line.strip() for line in entry_lines if line.strip()]
        if not lines:
            errors.append(f"{current_id} on page {current_page}: dish name missing")
            return
        entry = "\n".join(entry_lines)
        price_match = PRICE.search(entry)
        allergens, possible = _allergen_lists(entry)
        try:
            dishes.append(Dish(
                dish_id=current_id, restaurant_name=RESTAURANT_NAME,
                name=lines[0], description=_clean_description(lines[1:]),
                price=float(price_match.group(1).replace(",", "")) if price_match else None,
                cuisine=cuisine, category=category, ingredients=[], **_dietary_data(entry),
                allergens=allergens, may_contain_allergens=possible,
                dietary_source="Menu PDF", source_reference=f"{pdf_path.name}, page {current_page}",
                page_number=current_page,
            ))
        except ValidationError as exc:
            errors.append(f"{current_id} on page {current_page}: {exc}")

    for page_number, page in enumerate(reader.pages, start=1):
        for raw_line in (page.extract_text() or "").splitlines():
            line = raw_line.strip()
            if line in CUISINES or line in NON_CUISINE_HEADINGS:
                save_entry(); current_id = None; entry_lines = []
                cuisine = CUISINES.get(line)
                category = None
                continue
            if line in CATEGORIES:
                save_entry(); current_id = None; entry_lines = []
                category = line
                continue
            match = DISH_ID.match(line)
            if match:
                save_entry()
                current_id = f"D{match.group(1)}"
                current_page = page_number
                entry_lines = []
            elif current_id:
                entry_lines.append(line)
    save_entry()

    if errors:
        raise ValueError("Could not validate all menu entries:\n- " + "\n- ".join(errors))
    ids = [dish.dish_id for dish in dishes]
    if len(ids) != len(set(ids)):
        raise ValueError("The PDF contains duplicate dish IDs; fix the menu before indexing.")
    if not dishes:
        raise ValueError("No dish IDs like D001 were found in the PDF text. The PDF may need OCR.")
    return dishes
