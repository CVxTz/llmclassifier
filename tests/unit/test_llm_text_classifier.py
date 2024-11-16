import pytest

from llmclassifier.llm_clients import llm_openai_client
from llmclassifier.llm_text_classifier import LLMTextClassifier


def test_llm_text_classifier():
    # Define the categories and create the classifier
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(llm_client=llm_openai_client, categories=categories)

    # Test the classify method with a sample text
    text = "You won't believe what happened next! Watch for more"
    result = classifier.predict(text)

    # Assert that the result is as expected
    assert result == "clickbait"

    # Test another sample text
    text = "Donald trump won michigan"
    result = classifier.predict(text)

    # Assert that the result is as expected
    assert result == "news"


def test_llm_text_classifier_fit():
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(llm_client=llm_openai_client, categories=categories)

    texts = ["Donald trump won michigan", "You won't believe what happened next!"]
    labels = ["news", "clickbait"]

    classifier.fit(texts, labels)
    assert len(classifier.examples) == 2
    assert classifier.examples[0].page_content == "Donald trump won michigan"
    assert classifier.examples[0].metadata["label"] == "news"
    assert (
        classifier.examples[1].page_content == "You won't believe what happened next!"
    )
    assert classifier.examples[1].metadata["label"] == "clickbait"


def test_llm_text_classifier_fetch_examples():
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(llm_client=llm_openai_client, categories=categories)

    texts = ["Donald trump won michigan", "You won't believe what happened next!"]
    labels = ["news", "clickbait"]

    classifier.fit(texts, labels)

    # Test fetching examples when retriever is None
    examples = classifier.fetch_examples("Donald trump won michigan")
    assert len(examples) == 2
    assert examples[0].page_content == "Donald trump won michigan"
    assert examples[0].metadata["label"] == "news"
    assert examples[1].page_content == "You won't believe what happened next!"
    assert examples[1].metadata["label"] == "clickbait"


def test_llm_text_classifier_fetch_examples_chroma():
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(
        llm_client=llm_openai_client, categories=categories, max_examples=1
    )

    texts = ["Donald trump won michigan", "You won't believe what happened next!"]
    labels = ["news", "clickbait"]

    classifier.fit(texts, labels)

    # Test fetching examples when retriever is None
    examples = classifier.fetch_examples("Donald trump won")
    assert len(examples) == 1
    assert examples[0].page_content == "Donald trump won michigan"
    assert examples[0].metadata["label"] == "news"


def test_llm_text_classifier_classify():
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(
        llm_client=llm_openai_client, categories=categories, max_examples=1
    )

    texts = ["Donald trump won michigan", "You won't believe what happened next!"]
    labels = ["news", "clickbait"]

    classifier.fit(texts, labels)

    text = "Donald trump won florida"
    result = classifier.predict(text)

    assert result == "news"
    assert classifier.retriever is not None


def test_llm_text_classifier_invalid_input():
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(llm_client=llm_openai_client, categories=categories)

    # Test with empty text
    with pytest.raises(AssertionError):
        classifier.predict("")

    # Test with mismatched texts and labels
    with pytest.raises(AssertionError):
        classifier.fit(["Donald trump won michigan"], ["news", "clickbait"])

    # Test with empty texts
    with pytest.raises(AssertionError):
        classifier.fit([], ["news"])

    # Test with labels not in categories
    with pytest.raises(AssertionError):
        classifier.fit(["Donald trump won michigan"], ["politics"])
