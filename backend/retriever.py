from Bio import Entrez
from Bio import Medline

from backend.document import Document



# Required by NCBI
Entrez.email = "chnslalitha@gmail.com"


def search_pubmed(query: str, max_results: int = 5):
    """
    Search PubMed and return a list of PMIDs.
    """

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )

    results = Entrez.read(handle)
    handle.close()

    return results["IdList"]


def fetch_articles(pmids):
    """
    Fetch article metadata from PubMed.
    """

    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="medline",
        retmode="text"
    )

    records = Medline.parse(handle)

    documents = []

    for record in records:

        document = Document(
            pmid=record.get("PMID", ""),
            title=record.get("TI", ""),
            abstract=record.get("AB", ""),
            journal=record.get("JT", ""),
            year=record.get("DP", ""),
            authors=record.get("AU", [])
        )

        documents.append(document)

    handle.close()

    return documents


def retrieve_evidence(query: str):
    pmids = search_pubmed(query)
    return fetch_articles(pmids)