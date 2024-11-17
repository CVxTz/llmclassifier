import base64
from typing import Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from llmclassifier.data_model import (
    generate_multi_class_classification_model,
    generate_multi_label_classification_model,
)


class LLMMultiModalClassifier:
    def __init__(
        self,
        llm_client: BaseChatModel,
        categories: list[str],
        system_prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["categories", "schema"],
            template="Classify the following content into one of the following classes: {categories}.\n "
            "Use the following schema: {schema}",
        ),
        multi_label: bool = False,
    ):
        assert set(
            system_prompt_template.input_variables
        ).issubset(
            {"categories", "schema"}
        ), "System prompt template should be included in the following input variables: categories, schema"
        self.categories = categories
        if multi_label:
            self.categories_model = generate_multi_label_classification_model(
                categories
            )
        else:
            self.categories_model = generate_multi_class_classification_model(
                categories
            )

        self.system_prompt_template = system_prompt_template
        self.system_prompt = system_prompt_template.format(
            categories=categories, schema=self.categories_model.model_json_schema()
        )

        self.llm_classifier = llm_client.with_structured_output(self.categories_model)

    def predict(
        self, text: Optional[str] = None, image_url: Optional[str] = None
    ) -> list:
        assert text or image_url, "text or image url should be non-empty"

        content = []

        if text:
            content.append({"type": "text", "text": text})

        if image_url:
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )

        prediction = self.llm_classifier.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=content)]
        )

        return prediction.category


if __name__ == "__main__":
    import json
    from pathlib import Path

    from llmclassifier.llm_clients import llm_google_client

    with open(
        Path(__file__).parents[1] / "data" / "example_categories_images.json"
    ) as f:
        _categories = json.load(f)

    classifier = LLMMultiModalClassifier(
        llm_client=llm_google_client, categories=_categories, multi_label=True
    )

    unsplash_url = "https://images.unsplash.com/photo-1551205379-24fa06ec1928"
    resize_arg = "?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=400&fit=max&ixid=eyJhcHBfaWQiOjYyMzc5fQ"
    print(unsplash_url + resize_arg)
    _text = "black and white coated dog"

    print(classifier.predict(text=" ", image_url=unsplash_url + resize_arg))
