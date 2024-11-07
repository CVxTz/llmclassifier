from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from llmclassifier.data_model import generate_classification_model
from llmclassifier.llm_clients import llm_medium


class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]


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
        max_examples: int = 5,
    ):
        assert set(
            system_prompt_template.input_variables
        ).issubset(
            {"categories", "schema"}
        ), "System prompt template should be included in the following input variables: categories, schema"
        self.categories = categories
        self.categories_model = generate_classification_model(categories)
        self.system_prompt_template = system_prompt_template
        self.system_prompt = system_prompt_template.format(
            categories=categories, schema=self.categories_model.model_json_schema()
        )

        self.llm_classifier = llm_client.with_structured_output(self.categories_model)

        self.max_examples = max_examples

        self.examples = None
        self.vector_store = None
        self.retriever = None

    def predict(self, text: str) -> str:
        assert text, "text should be non-empty"
        messages = [
            SystemMessage(content=self.system_prompt),
        ]

        for example in self.fetch_examples(text=text):
            messages.append(HumanMessage(content=example.page_content))
            messages.append(AIMessage(content=example.metadata["label"]))

        messages.append(HumanMessage(content=text))
        prediction = self.llm_classifier.invoke(messages)

        return prediction.category

    def fit(self, texts, labels):
        assert set(labels).issubset(
            set(self.categories)
        ), "Train labels should be included in the categories used in the init"
        assert len(texts) == len(labels), "Texts and labels should be equal"
        assert len(texts), "Train texts should be non-empty"

        self.examples = [
            Document(page_content=text, metadata={"label": label})
            for text, label in zip(texts, labels)
        ]

        if len(self.examples) <= self.max_examples:
            return self

        else:
            # Add to vectorDB
            self.vector_store = Chroma.from_documents(
                documents=self.examples,
                collection_name="llm-classifier",
                embedding=ChromaEmbeddingsAdapter(
                    embedding_functions.DefaultEmbeddingFunction()
                ),
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.max_examples}
            )

    def fetch_examples(self, text: str) -> list[Document]:
        if self.retriever is not None:
            return self.retriever.invoke(text)

        elif self.examples is not None:
            return self.examples[: self.max_examples]

        else:
            return []


if __name__ == "__main__":
    classifier = LLMTextClassifier(categories=["news", "clickbait"])

    print(
        classifier.predict("You won't believe what happened next! Watch for more")
    )  # returns clickbait
    print(classifier.predict("Donald trump won michigan"))  # returns news
