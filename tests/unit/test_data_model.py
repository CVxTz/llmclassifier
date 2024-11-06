import pytest
from pydantic import BaseModel, ValidationError

from llmclassifier.data_model import generate_classification_model


def test_generate_classification_model():
    # Test with valid input
    Categories = generate_classification_model(["Yes", "No"])

    # Check that the model is correctly generated
    assert issubclass(Categories, BaseModel)

    # Check that the model validates the input correctly
    categories = Categories(category="Yes")
    assert categories.category == "Yes"

    # Check that the model raises a ValidationError for invalid input
    with pytest.raises(ValidationError):
        Categories(category="Maybe")


def test_generate_classification_model_empty_list():
    # Test with an empty list, which should raise an AssertionError
    with pytest.raises(AssertionError):
        generate_classification_model([])
