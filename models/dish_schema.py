"""Validated data model for one dish in the MenuLens PDF."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Dish(BaseModel):
    """Menu facts. None means the PDF does not establish the value."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    dish_id: str = Field(min_length=1)
    restaurant_name: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    price: float | None = Field(default=None, ge=0)
    cuisine: str | None = None
    category: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    vegetarian: bool | None = None
    vegan: bool | None = None
    jain: bool | None = None
    gluten_free: bool | None = None
    dairy_free: bool | None = None
    egg_free: bool | None = None
    dietary_conditions: list[str] = Field(default_factory=list)
    allergens: list[str] | None = None
    may_contain_allergens: list[str] | None = None
    dietary_source: str | None = None
    source_reference: str | None = None
    page_number: int | None = Field(default=None, ge=1)

    @field_validator("ingredients", "dietary_conditions", mode="before")
    @classmethod
    def parse_list_values(cls, value: object) -> object:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("allergens", "may_contain_allergens", mode="before")
    @classmethod
    def keep_unknown_allergens_unknown(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            parsed = [part.strip() for part in value.split(",") if part.strip()]
            return parsed or None
        return value

    @field_validator(
        "description", "cuisine", "category", "dietary_source", "source_reference", mode="before"
    )
    @classmethod
    def blank_optional_strings_are_unknown(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value
