from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from llmclassifier.data_model import generate_classification_model
from llmclassifier.llm_clients import llm_medium


class LLMTextClassifier:
    def __init__(
        self,
        categories: list[str],
        system_prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["categories", "schema"],
            template="Classify the following text into one of the following classes: {categories}.\n "
            "Use the following schema: {schema}",
        ),
        llm_client: BaseChatModel = llm_medium,
    ):
        assert (
            set(system_prompt_template.input_variables) == {"categories", "schema"}
        ), "System prompt template should have these following input variables: categories, schema"
        self.categories = categories
        self.categories_model = generate_classification_model(categories)
        self.system_prompt_template = system_prompt_template
        self.system_prompt = system_prompt_template.format(
            categories=categories, schema=self.categories_model.model_json_schema()
        )

        self.llm_classifier = llm_client.with_structured_output(self.categories_model)

    def classify(self, text: str):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=text),
        ]
        prediction = self.llm_classifier.invoke(messages)

        return prediction.category


if __name__ == "__main__":
    classifier = LLMTextClassifier(categories=["news", "clickbait"])

    print(
        classifier.classify("You won't believe what happened next! Watch for more")
    )  # returns clickbait
    print(classifier.classify("Donald trump won michigan"))  # returns news
