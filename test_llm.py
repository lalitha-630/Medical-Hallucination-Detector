from backend.llm import generate_response
from backend.retriever import retrieve_evidence
from backend.chunker import chunk_documents
from backend.embeddings import embed_chunks
from backend.vector_store import VectorStore
from backend.verifier import verify_claim
from backend.correction import generate_corrected_response


def main():

    question = input("Enter your medical question: ")

    result = generate_response(question)

    print("\n========== LLM RESPONSE ==========\n")
    print(result["answer"])

    print("\n========== TOP 5 CLAIMS ==========\n")

    verification_results = []

    for i, item in enumerate(result["claims"], start=1):

        claim = item["claim"]
        query = item["query"]

        print("=" * 80)
        print(f"CLAIM {i}")
        print(claim)
        print(f"\nPubMed Query: {query}")

        # -------------------------------------------------
        # Retrieve PubMed papers
        # -------------------------------------------------
        documents = retrieve_evidence(query)

        print(f"\nRetrieved {len(documents)} papers\n")

        for j, doc in enumerate(documents, start=1):
            print(f"Paper {j}")
            print(f"PMID    : {doc.pmid}")
            print(f"Title   : {doc.title}")
            print(f"Journal : {doc.journal}")
            print(f"Year    : {doc.year}")
            print("-" * 70)

        # -------------------------------------------------
        # Chunking
        # -------------------------------------------------
        chunks = chunk_documents(documents)

        print(f"\nGenerated {len(chunks)} chunks")

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------
        embeddings, chunks = embed_chunks(chunks)

        print(f"Embeddings Shape : {embeddings.shape}")

        # -------------------------------------------------
        # FAISS Vector Store
        # -------------------------------------------------
        vector_store = VectorStore(embeddings, chunks)

        # -------------------------------------------------
        # Retrieve Top Relevant Chunks
        # -------------------------------------------------
        results = vector_store.search(claim, top_k=3)

        print("\n========== TOP 3 RELEVANT CHUNKS ==========\n")

        for rank, (chunk, score) in enumerate(results, start=1):

            print("=" * 70)
            print(f"Rank       : {rank}")
            print(f"Similarity : {score:.4f}")
            print(f"Document   : {chunk.document_id}")
            print(f"Chunk ID   : {chunk.chunk_id}")
            print("\nChunk Text:\n")
            print(chunk.text)
            print()

        # -------------------------------------------------
        # DeBERTa Verification
        # -------------------------------------------------
        evidence_chunks = [chunk for chunk, _ in results]

        verification = verify_claim(claim, evidence_chunks)

        verification["claim"] = claim
        verification_results.append(verification)

        print("\n========== CLAIM VERIFICATION ==========\n")

        print(f"Claim      : {claim}")
        print(f"Verdict    : {verification['verdict']}")
        print(f"Confidence : {verification['confidence'] * 100:.2f}%")
        print(f"Document   : {verification['document_id']}")
        print(f"Chunk ID   : {verification['chunk_id']}")

        print("\nClass Probabilities")
        print("-" * 50)
        print(
            f"Supported      : {verification['scores']['supported'] * 100:.2f}%"
        )
        print(
            f"Contradicted   : {verification['scores']['contradicted'] * 100:.2f}%"
        )
        print(
            f"Neutral        : {verification['scores']['neutral'] * 100:.2f}%"
        )

        print("\nEvidence Used")
        print("-" * 50)
        print(verification["evidence"])

        print("\n" + "=" * 80 + "\n")

    # -------------------------------------------------
    # Generate Corrected Response
    # -------------------------------------------------
    correction = generate_corrected_response(
        original_answer=result["answer"],
        verification_results=verification_results
    )

    print("\n========== CORRECTED RESPONSE ==========\n")
    print(correction["corrected_response"])

    print("\n========== CORRECTED CLAIMS ==========\n")

    if correction["corrected_claims"]:

        for item in correction["corrected_claims"]:

            print("=" * 70)
            print(f"Claim      : {item['claim']}")
            print(f"Verdict    : {item['verdict']}")
            print(f"Confidence : {item['confidence']}%")

            print("\nEvidence:")
            print(item["evidence"])
            print()

    else:

        print("All claims were verified. No correction required.")


if __name__ == "__main__":
    main()