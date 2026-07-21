from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    authors: List[str]