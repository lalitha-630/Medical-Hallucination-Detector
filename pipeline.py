from backend.llm import generate_response
from backend.retriever import retrieve_evidence
from backend.chunker import chunk_documents
from backend.embeddings import embed_chunks
from backend.vector_store import VectorStore
from backend.verifier import verify_claim
from backend.correction import generate_corrected_response


def run_pipeline(question):

    # -----------------------------
    # Generate LLM Response
    # -----------------------------
    print("Generating LLM response...")
    result = generate_response(question)

    verification_results = []

    for item in result["claims"]:

        claim = item["claim"]
        query = item["query"]

        # Retrieve PubMed papers
        print("Retrieving evidence...")
        documents = retrieve_evidence(query)

        # Chunk papers
        print("Chunking...")
        chunks = chunk_documents(documents)

        # Generate embeddings
        print("Embedding...")
        embeddings, chunks = embed_chunks(chunks)

        # Create Vector Store
        print("Creating Vector Store...")
        vector_store = VectorStore(embeddings, chunks)

        # Retrieve top evidence
        print("Searching...")
        results = vector_store.search(claim, top_k=3)

        evidence_chunks = [chunk for chunk, _ in results]

        # Verify claim
        print("Verifying...")
        verification = verify_claim(claim, evidence_chunks)

        verification["claim"] = claim

        verification_results.append(verification)

    # -----------------------------
    # Generate Corrected Response
    # -----------------------------
    print("Generating correction...")
    correction = generate_corrected_response(
        original_answer=result["answer"],
        verification_results=verification_results
    )

    # -----------------------------
    # Summary
    # -----------------------------
    supported = sum(
        1 for v in verification_results
        if v["verdict"] == "Supported"
    )

    contradicted = sum(
        1 for v in verification_results
        if v["verdict"] == "Contradicted"
    )

    not_enough = sum(
        1 for v in verification_results
        if v["verdict"] == "Not Enough Evidence"
    )

    return {

        "question": question,

        "answer": result["answer"],

        "claims": verification_results,

        "corrected_response":
            correction["corrected_response"],

        "corrected_claims":
            correction["corrected_claims"],

        "summary": {

            "total": len(verification_results),

            "supported": supported,

            "contradicted": contradicted,

            "not_enough": not_enough

        }

    }