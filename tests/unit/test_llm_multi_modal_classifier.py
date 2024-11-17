import json
from pathlib import Path

from llmclassifier.llm_clients import llm_google_client
from llmclassifier.llm_multi_modal_classifier import LLMMultiModalClassifier


def test_llm_multi_modal_classifier():
    with open(
        Path(__file__).parents[2] / "data" / "example_categories_images.json"
    ) as f:
        categories = json.load(f)
    classifier = LLMMultiModalClassifier(
        llm_client=llm_google_client, categories=categories, multi_label=True
    )

    unsplash_url = "https://images.unsplash.com/photo-1551205379-24fa06ec1928"
    resize_arg = "?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=400&fit=max&ixid=eyJhcHBfaWQiOjYyMzc5fQ"
    print(unsplash_url + resize_arg)
    _text = "black and white coated dog"

    keywords = classifier.predict(text=" ", image_url=unsplash_url + resize_arg)

    assert isinstance(keywords, list)
    assert keywords
    assert "dog" in keywords
