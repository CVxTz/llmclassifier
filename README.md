# llmclassifier

`llmclassifier` is a Python library for building reliable and accurate text classification pipelines using large language models (LLMs). This library includes techniques for constrained generation, few-shot prompting, and dynamic few-shot selection, allowing users to leverage LLMs for text classification tasks without the need to train a custom model from scratch. It’s designed to be flexible, easy to use, and compatible with various OpenAI and LangChain models.

## Features

- **Constrained Generation**: Limit the LLM's output to predefined classes, reducing post-processing needs.
- **Few-Shot Prompting**: Improve accuracy by providing example input-output pairs in the prompt context.
- **Dynamic Few-Shot Selection**: Use ChromaDB to select contextually relevant examples for each query, maximizing classification performance.

## Installation

To install `llmclassifier`, you can use pip with the following command:

```bash
pip install git+https://github.com/CVxTz/llmclassifier.git#egg=llmclassifier
```

## Usage

The following example demonstrates how to use `llmclassifier` to set up an LLM-based text classification model, fit it with training data, and use it to classify new text inputs.

### Example Code

```python
import os
from dotenv import load_dotenv
from llmclassifier import LLMTextMultiClassClassifier

# Load environment variables containing API credentials
load_dotenv()

categories = ["news", "clickbait"]
classifier = LLMTextMultiClassClassifier(categories=categories, max_examples=1)

texts = ["Donald trump won michigan", "You won't believe what happened next!"]
labels = ["news", "clickbait"]

classifier.fit(texts, labels)

text = "Donald trump won florida"
result = classifier.predict(text)

```

### Explanation of Code Components

1. **Initialize the Classifier**: Define the categories for classification and create an instance of `LLMTextClassifier`.
2. **Fit the Classifier**: Provide a set of labeled examples to allow the classifier to fine-tune its predictions.
3. **Classify Text**: Use the `predict` method to classify new input text, based on the pre-defined categories.

## Available Techniques

- **Constrained Generation**: Ensures that LLM output matches one of the predefined categories, helping to prevent unwanted or ambiguous responses.
- **Few-Shot Prompting**: Leverages a few labeled examples to set expectations for the LLM, improving the quality of predictions.
- **Dynamic Few-Shot Selection**: Uses ChromaDB to retrieve relevant examples for each query, allowing the LLM to perform contextually-aware predictions.

## Requirements

- **Python 3.9+**

## Environment Variables

To use `llmclassifier` with OpenAI-compatible models, ensure you have the following environment variables set:

```bash
export LLM_BASE_URL=<Your LLM API Base URL>
export OPENAI_API_KEY=<Your API Key>
export OPENAI_MODEL_NAME=modelname1
```

Alternatively, you can set these variables in a `.env` file, and load it with `dotenv`.

## Contributing

Contributions are welcome! Feel free to submit pull requests or report issues to improve the functionality and usability of `llmclassifier`.


## Datasets

https://github.com/unsplash/datasets/blob/master/how-to/python/pandas.ipynb 