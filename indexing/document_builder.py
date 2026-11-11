"""Create one LlamaIndex document per validated dish."""

from llama_index.core import Document

from models.dish_schema import Dish


def dish_to_document(dish: Dish) -> Document:
    lines = [f"Dish ID: {dish.dish_id}", f"Dish: {dish.name}"]
    lines.append(f"Restaurant: {dish.restaurant_name}")
    if dish.description:
        lines.append(f"Description: {dish.description}")
    if dish.price is not None:
        lines.append(f"Price: ₹{dish.price:g}")
    if dish.cuisine:
        lines.append(f"Cuisine: {dish.cuisine}")
    if dish.vegetarian is not None:
        lines.append(f"Vegetarian: {dish.vegetarian}")
    if dish.vegan is not None:
        lines.append(f"Vegan: {dish.vegan}")
    if dish.allergens is not None:
        lines.append(f"Allergen note: {', '.join(dish.allergens)}")
    else:
        lines.append("Allergen information: not specified in the menu")

    metadata = {
        "dish_id": dish.dish_id,
        "restaurant_name": dish.restaurant_name,
        "name": dish.name,
        "price": dish.price,
        "cuisine": dish.cuisine,
        "vegetarian": dish.vegetarian,
        "vegan": dish.vegan,
        "allergens": ", ".join(dish.allergens) if dish.allergens is not None else None,
        "dietary_source": dish.dietary_source,
        "source_reference": dish.source_reference,
        "page_number": dish.page_number,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return Document(text="\n".join(lines), metadata=metadata, id_=dish.dish_id)


def build_documents(dishes: list[Dish]) -> list[Document]:
    return [dish_to_document(dish) for dish in dishes]
