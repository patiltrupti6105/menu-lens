"""Read-only retrieval tools with validated inputs."""

from langchain_core.tools import tool

from models.query_schema import CompareDishesInput, DishLookupInput, SearchDishesInput
from retrieval.search import format_hit, search_menu


@tool(args_schema=SearchDishesInput)
def search_dishes(
    query: str,
    cuisine: str | None = None,
    max_price: float | None = None,
    vegetarian: bool | None = None,
    vegan: bool | None = None,
    restaurant_name: str | None = None,
    top_k: int = 5,
) -> str:
    """Find dishes using semantic search and verified structured filters."""
    filters = {
        key: value
        for key, value in {
            "cuisine": cuisine,
            "max_price": max_price,
            "vegetarian": vegetarian,
            "vegan": vegan,
            "restaurant_name": restaurant_name,
        }.items()
        if value is not None
    }
    hits = search_menu(query, top_k, filters)
    if not hits:
        return "No dishes in the menu match those search terms and verified filters. Unknown dietary values do not count as a match."
    return "\n\n".join(format_hit(hit) for hit in hits)


@tool(args_schema=DishLookupInput)
def get_dish_details(query: str) -> str:
    """Look up menu text about a dish and return its PDF page source."""
    hits = search_menu(query, 3)
    return "\n\n".join(format_hit(hit) for hit in hits) if hits else "No supporting menu text was found."


@tool(args_schema=CompareDishesInput)
def compare_dishes(dish_names: list[str]) -> str:
    """Retrieve menu passages for two or more requested dishes for a grounded comparison."""
    sections = []
    for name in dish_names:
        hits = search_menu(name, 2)
        sections.append(f"Requested dish: {name}\n" + (
            "\n\n".join(format_hit(hit) for hit in hits)
            if hits else "No supporting menu text was found."
        ))
    return "\n\n---\n\n".join(sections)


MENU_TOOLS = [search_dishes, get_dish_details, compare_dishes]
