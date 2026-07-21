import faiss
import numpy as np

from backend.embeddings import embed_text


class VectorStore:

    def __init__(self, embeddings, chunks):
        """
        embeddings : np.ndarray of shape (N, D)
        chunks     : List[Chunk]
        """

        self.chunks = chunks

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings.astype(np.float32))

    def search(self, claim, top_k=3):
        """
        Retrieve the most relevant chunks for a claim.

        Args:
            claim (str)
            top_k (int)

        Returns:
            List[(Chunk, similarity_score)]
        """

        claim_embedding = embed_text(claim).astype(np.float32)

        claim_embedding = np.expand_dims(claim_embedding, axis=0)

        scores, indices = self.index.search(claim_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append((self.chunks[idx], float(score)))

        return results