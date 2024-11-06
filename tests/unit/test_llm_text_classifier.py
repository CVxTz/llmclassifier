from llmclassifier.llm_text_classifier import LLMTextClassifier


def test_llm_text_classifier():
    # Define the categories and create the classifier
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(categories=categories)

    # Test the classify method with a sample text
    text = "You won't believe what happened next! Watch for more"
    result = classifier.classify(text)

    # Assert that the result is as expected
    assert result == "clickbait"

    # Test another sample text
    text = "Donald trump won michigan"
    result = classifier.classify(text)

    # Assert that the result is as expected
    assert result == "news"
