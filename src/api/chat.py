from src.services.rag.ra import RA

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse, JSONResponse
import json


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/answer", response_model=dict)
async def answer(question: str = Body()):
    context = RA.answer_question(question=question)
    answer = context.answer
    data = {
        "context": context,
        "answer": answer,
        "model": RA.retriever.embedding_model_string
    }
    return JSONResponse(content=data)


@router.post("/answer/v2/stream")
async def answer_v2_stream(question: str = Body()):
    async def generate_stream():
        for chunk in RA.retriever.retrieve_stream(question):
            # yield f"data: {chunk}\n\n"
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
