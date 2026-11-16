"""Create one searchable LlamaIndex document for each validated dish."""

from llama_index.core import Document

from models.dish_schema import Dish


def dish_to_document(dish: Dish) -> Document:
    lines = [f"Dish ID: {dish.dish_id}", f"Dish: {dish.name}"]
    if dish.category:
        lines.append(f"Category: {dish.category}")
    if dish.description:
        lines.append(f"Description: {dish.description}")
    if dish.price is not None:
        lines.append(f"Price: ₹{dish.price:g}")
    for field in ("cuisine", "vegetarian", "vegan", "jain", "gluten_free", "dairy_free", "egg_free"):
        value = getattr(dish, field)
        if value is not None:
            lines.append(f"{field.replace('_', ' ').title()}: {value}")
    if dish.dietary_conditions:
        lines.append("Dietary conditions: " + "; ".join(dish.dietary_conditions))
    if dish.allergens is not None:
        lines.append("Contains allergens: " + ", ".join(dish.allergens))
    if dish.may_contain_allergens is not None:
        lines.append("May contain allergens: " + ", ".join(dish.may_contain_allergens))
    if dish.ingredients:
        lines.append("Ingredients: " + ", ".join(dish.ingredients))
    if dish.allergens is None and dish.may_contain_allergens is None:
        lines.append("Allergen information: not specified in the menu")

    metadata = {
        "dish_id": dish.dish_id, "restaurant_name": dish.restaurant_name, "name": dish.name,
        "description": dish.description, "price": dish.price, "cuisine": dish.cuisine,
        "category": dish.category, "vegetarian": dish.vegetarian, "vegan": dish.vegan,
        "jain": dish.jain, "gluten_free": dish.gluten_free, "dairy_free": dish.dairy_free,
        "egg_free": dish.egg_free, "dietary_conditions": " | ".join(dish.dietary_conditions),
        "ingredients": " | ".join(dish.ingredients),
        "allergens": " | ".join(dish.allergens) if dish.allergens is not None else None,
        "may_contain_allergens": " | ".join(dish.may_contain_allergens) if dish.may_contain_allergens is not None else None,
        "dietary_source": dish.dietary_source, "source_reference": dish.source_reference,
        "page_number": dish.page_number,
    }
    # Chroma accepts scalar metadata only; absent values remain absent (and unknown).
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return Document(text="\n".join(line for line in lines if line), metadata=metadata, id_=dish.dish_id)


def build_documents(dishes: list[Dish]) -> list[Document]:
    return [dish_to_document(dish) for dish in dishes]
