from backend.chunk import Chunk


def chunk_documents(documents, chunk_size=200):
    """
    Split document abstracts into smaller chunks.
    """

    chunks = []

    for doc in documents:

        words = doc.abstract.split()

        chunk_id = 1

        for i in range(0, len(words), chunk_size):

            text = " ".join(words[i:i + chunk_size])

            chunk = Chunk(
                document_id=doc.pmid,
                chunk_id=chunk_id,
                text=text
            )

            chunks.append(chunk)

            chunk_id += 1

    return chunks