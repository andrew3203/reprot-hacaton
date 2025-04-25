from src.services.rag.retrieval_augmentation import RetrievalAugmentationConfig
from src.services.rag.retrieval_augmentation import RetrievalAugmentation
from src.services.rag.retrievers import GEMMASummarizationModel, GEMMAQAModel, SBertEmbeddingModel
from src.core.config import settings
import os

RAC = RetrievalAugmentationConfig(
    summarization_model=GEMMASummarizationModel(),
    qa_model=GEMMAQAModel(),
    embedding_model=SBertEmbeddingModel(),
)

RA = RetrievalAugmentation(config=RAC, tree=settings.tree)

if os.path.exists("data/info"):
    RA.load("data/info")
else:
    # Initialize with empty data to create the retriever
    RA.add_documents("")
