from src.llm import ChatCompletionRequest, GenAIAgentClient, Message
from src import (
    BaseSummarizationModel,
    BaseQAModel,
    BaseEmbeddingModel,
    RetrievalAugmentationConfig,
)
from src import RetrievalAugmentation
from src.config import settings
from sentence_transformers import SentenceTransformer
import uvicorn
from fastapi import FastAPI, Body
import os

client = GenAIAgentClient(base_url=settings.BASE_ULR, api_key=settings.API_KEY)


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

        response = client.chat_completion(payload=request_payload)
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

        response = client.chat_completion(payload=request_payload)
        return response["choices"][0]["message"]["content"]


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


RAC = RetrievalAugmentationConfig(
    summarization_model=GEMMASummarizationModel(),
    qa_model=GEMMAQAModel(),
    embedding_model=SBertEmbeddingModel(),
)
if os.path.exists("data/info"):
    tree = "data/info"
else:
    tree = None
    
RA = RetrievalAugmentation(config=RAC, tree=tree)

docs: list[str] = []
app = FastAPI()


@app.post("/upload", response_model=str)
async def add_doc(data: str = Body()):
    docs.append(data)
    return "ok"


@app.get("/index", response_model=dict)
async def index_docs():
    text = "\n\n".join(docs)
    RA.add_documents(text)
    RA.save("data/info")
    return "ok"


@app.post("/answer", response_model=str)
async def answer(question: str = Body()):
    return RA.answer_question(question=question)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=10000,
    )
