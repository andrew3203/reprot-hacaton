from abc import ABC, abstractmethod
from src.services.chat.llm import ChatCompletionRequest, Message
from src.services.chat.summarization_models import BaseSummarizationModel
from src.services.chat.qa_models import BaseQAModel
from src.services.chat.embedding_models import BaseEmbeddingModel
from src.core.config import settings
from sentence_transformers import SentenceTransformer


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> str:
        pass


class GEMMASummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name=""):
        pass

    def summarize(self, context, max_tokens=150):
        request_payload = ChatCompletionRequest(
            messages=[
                Message(
                    role="user",
                    content=f"Write a summary of the following, including as many key details as possible: {context}:",
                )
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=3000,
            max_completion_tokens=400,
        )

        response = settings.client.chat_completion(payload=request_payload)
        return response["choices"][0]["message"]["content"]


class GEMMAQAModel(BaseQAModel):
    def __init__(self, model_name=""):
        pass

    def answer_question(self, context, question):
        request_payload = ChatCompletionRequest(
            messages=[
                Message(
                    role="user",
                    content=f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                )
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=3000,
            max_completion_tokens=400,
        )

        response = settings.client.chat_completion(payload=request_payload)
        return response["choices"][0]["message"]["content"]


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
