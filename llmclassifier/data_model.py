from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


def generate_multi_class_classification_model(list_classes: list[str]):
    assert list_classes

    class ClassificationOutput(BaseModel):
        category: Literal[tuple(list_classes)]

    return ClassificationOutput


def generate_multi_label_classification_model(list_classes: list[str]):
    assert list_classes

    class ClassificationOutput(BaseModel):
        category: List[Literal[tuple(list_classes)]] = Field(default_factory=list)

    @field_validator('category', mode='before')
    def filter_invalid_categories(cls, value):
        if isinstance(value, list):  # Ensure input is a list
            return [v for v in value if v in list_classes]
        return []  # Return an empty list if the input is not a list

    return ClassificationOutput


if __name__ == "__main__":
    Categories = generate_multi_class_classification_model(["Yes", "No"])

    categories = Categories(category="Yes")

    print(categories)
