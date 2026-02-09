
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# load environment variables

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

#load model
_embedding_model=SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str])-> np.ndarray:

    """
    Generate embeddings for list of text
    """
    if not texts:
        return np.array([])

    embeddings=_embedding_model.encode(
        texts,
        batch_size=32,show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings