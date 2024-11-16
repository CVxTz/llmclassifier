import concurrent.futures

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from llmclassifier import LLMTextClassifier
from llmclassifier.llm_clients import llm_openai_client

if __name__ == "__main__":
    test_size = 300
    max_examples = 5
    train = fetch_20newsgroups(subset="train")
    test = fetch_20newsgroups(subset="test")

    X_train = train.data
    Y_train = [train.target_names[i] for i in train.target]

    X_test = test.data[:test_size]
    Y_test = [test.target_names[i] for i in test.target][:test_size]

    categories = list(set(Y_train))

    classifier = LLMTextClassifier(
        categories=categories, max_examples=max_examples, llm_client=llm_openai_client
    )  # Zero shot
    classifier.fit(X_train[:max_examples], Y_train[:max_examples])  # Few shot
    # classifier.fit(X_train, Y_train)  # Dynamic Few shot

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        predictions = list(
            tqdm(executor.map(classifier.predict, X_test), total=len(X_test))
        )

    print(f"Accuracy: {accuracy_score(y_true=Y_test, y_pred=predictions)}")

    # Zero shot : 76.3%
    # Few shot : 79%
    # Dynamic Few shot : 89.3%
