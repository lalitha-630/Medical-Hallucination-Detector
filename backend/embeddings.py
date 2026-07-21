from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model only once
MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    """
    Generate an embedding for a single piece of text.

    Args:
        text (str): Input text.

    Returns:
        np.ndarray: Embedding vector.
    """
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding


def embed_chunks(chunks):
    """
    Generate embeddings for a list of Chunk objects.

    Args:
        chunks (List[Chunk])

    Returns:
        tuple:
            embeddings -> np.ndarray
            chunks -> Original Chunk objects
    """

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings, chunks