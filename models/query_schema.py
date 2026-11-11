"""Validated arguments for menu search and comparison tools."""

from pydantic import BaseModel, Field


class SearchDishesInput(BaseModel):
    query: str = Field(min_length=2, description="What the diner is looking for")
    cuisine: str | None = None
    max_price: float | None = Field(default=None, ge=0)
    vegetarian: bool | None = None
    vegan: bool | None = None
    restaurant_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class DishLookupInput(BaseModel):
    query: str = Field(min_length=2, description="Dish name or identifying details")


class CompareDishesInput(BaseModel):
    dish_names: list[str] = Field(min_length=2, max_length=5)
