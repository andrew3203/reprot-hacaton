from src.services.rag.ra import RA
from src.core.logger import logger
import os
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse


router = APIRouter(prefix="/files", tags=["Files"])

docs: list[str] = []


@router.post("/upload", response_model=str)
async def add_doc(data: str = Body()):
    docs.append(data)
    # Check if retriever exists and initialize if needed
    if not hasattr(RA, 'retriever') or RA.retriever is None:
        # Check if there's saved data to load
        try:
            RA.load("data/info")
        except Exception as e:
            logger.error(f"Error loading retriever: {e}")
            RA.add_documents("")
    res = {"status": "ok", "model": RA.retriever.context_embedding_model}
    return JSONResponse(content=res)


@router.get("/index", response_model=dict)
async def index_docs():
    text = "\n\n".join(docs)
    RA.add_documents(text)
    os.makedirs("data", exist_ok=True)
    RA.save("data/info")
    res = {"status": "ok", "model": RA.retriever.context_embedding_model}
    return JSONResponse(content=res)


@router.post("/upload/batch", response_model=dict)
async def add_docs(data: list = Body()):
    """Добавить несколько документов"""
    docs.extend(data)
    res = {"status": "ok", "model": RA.retriever.context_embedding_model}
    return JSONResponse(content=res)
