
import time

from embeddings.embedding_service import (
    generate_document_embedding,
)


def generate_embeddings_in_batches(
    chunks,
    batch_size=10,
    max_retries=3,
    retry_delay=2,
):
    """
    Generate embeddings for multiple document chunks.

    Each successful result retains its original chunk metadata.

    Failed chunks are recorded separately.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    results = []
    failures = []

    total_chunks = len(chunks)

    if total_chunks == 0:
        return {
            "results": results,
            "failures": failures,
        }

    total_batches = (
        total_chunks + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(0, total_chunks, batch_size),
        start=1,
    ):

        batch = chunks[start:start + batch_size]

        print(
            f"\nProcessing batch {batch_number}/{total_batches} "
            f"({len(batch)} chunks)"
        )

        for chunk in batch:

            chunk_id = chunk["chunk_index"]

            for attempt in range(max_retries + 1):

                try:

                    embedding = generate_document_embedding(
                        text=chunk["content"],
                        title=chunk["section_title"],
                    )

                    results.append({
                        **chunk,
                        "embedding": embedding,
                    })

                    print(
                        f"  Chunk {chunk_id}: embedding generated"
                    )

                    break

                except Exception as error:

                    if attempt < max_retries:

                        delay = retry_delay * (2 ** attempt)

                        print(
                            f"  Chunk {chunk_id}: attempt "
                            f"{attempt + 1} failed: {error}"
                        )

                        print(
                            f"  Retrying in {delay} seconds..."
                        )

                        time.sleep(delay)

                    else:

                        print(
                            f"  Chunk {chunk_id}: failed after "
                            f"{max_retries + 1} attempts."
                        )

                        failures.append({
                            "chunk_index": chunk_id,
                            "error": str(error),
                        })

    return {
        "results": results,
        "failures": failures,
    }
