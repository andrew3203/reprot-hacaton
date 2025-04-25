from src.services.rag.ra import RA

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/files", tags=["Files"])

docs: list[str] = []


@router.post("/upload", response_model=str)
async def add_doc(data: str = Body()):
    docs.append(data)
    res = {"status": "ok", "model": RA.retriever.embedding_model_string}
    return JSONResponse(content=res)


@router.get("/index", response_model=dict)
async def index_docs():
    text = "\n\n".join(docs)
    RA.add_documents(text)
    RA.save("data/info")
    res = {"status": "ok", "model": RA.retriever.embedding_model_string}
    return JSONResponse(content=res)


@router.post("/upload/batch", response_model=dict)
async def add_docs(data: list = Body()):
    """Добавить несколько документов"""
    docs.extend(data)
    res = {"status": "ok", "model": RA.retriever.embedding_model_string}
    return JSONResponse(content=res)
