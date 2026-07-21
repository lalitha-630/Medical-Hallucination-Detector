from dataclasses import dataclass


@dataclass
class Chunk:
    document_id: str
    chunk_id: int
    text: str