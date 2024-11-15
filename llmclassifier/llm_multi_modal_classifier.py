import base64
from typing import Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from llmclassifier.data_model import generate_multi_label_classification_model


class LLMMultiModalMultiLabelClassifier:
    def __init__(
        self,
        llm_client: BaseChatModel,
        categories: list[str],
        system_prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["categories", "schema"],
            template="Classify the following content into one of the following classes: {categories}.\n "
            "Use the following schema: {schema}",
        ),
    ):
        assert set(
            system_prompt_template.input_variables
        ).issubset(
            {"categories", "schema"}
        ), "System prompt template should be included in the following input variables: categories, schema"
        self.categories = categories
        self.categories_model = generate_multi_label_classification_model(categories)
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

    classifier = LLMMultiModalMultiLabelClassifier(
        llm_client=llm_google_client, categories=_categories
    )

    unsplash_url = "https://images.unsplash.com/photo-1546561927-370e6e533f9b"
    resize_arg = "?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=400&fit=max&ixid=eyJhcHBfaWQiOjYyMzc5fQ"
    print(unsplash_url + resize_arg)
    _text = (
        "So we were looking for urban explorations! After hours of shooting, we found this place who actually has "
        "a strange color palette! It mostly has a red hue, but the soft blue tones kill it completely! Also it’s "
        "the result of using multiple objects to recreate rainbows and create a colorful atmosphere. Of course,"
        " it also proves that location isn’t everything when you’re focused on transmitting concrete emotions to "
        "the viewer and self-expressing your art. Sincerely, one of my favorites for those brilliant tones and "
        "composition!"
    )

    print(classifier.predict(text=_text, image_url=unsplash_url + resize_arg))
