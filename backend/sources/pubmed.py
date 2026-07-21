from Bio import Entrez
from backend.query_optimizer import optimize_query

Entrez.email = "chnslalitha@gmail.com"


def search_pubmed(query, max_results=5):

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )

    record = Entrez.read(handle)
    handle.close()

    return record["IdList"]


def fetch_pubmed_articles(ids):

    if not ids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(ids),
        rettype="abstract",
        retmode="xml"
    )

    records = Entrez.read(handle)
    handle.close()

    evidence = []

    for article in records["PubmedArticle"]:

        try:

            article_info = article["MedlineCitation"]["Article"]

            title = str(article_info["ArticleTitle"])

            abstract = ""

            if "Abstract" in article_info:

                abstract = " ".join(
                    str(x)
                    for x in article_info["Abstract"]["AbstractText"]
                )

            pmid = article["MedlineCitation"]["PMID"]

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            evidence.append({

                "source": "PubMed",

                "title": title,

                "content": abstract,

                "url": url,

                "score": None

            })

        except Exception:
            continue

    return evidence


def retrieve_pubmed_evidence(claim):

    query = optimize_query(claim)

    ids = search_pubmed(query)

    return fetch_pubmed_articles(ids)