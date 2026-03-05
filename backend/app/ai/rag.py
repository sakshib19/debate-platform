"""
RAG module: Retrieval-Augmented Generation over debate manuals.

Handles:
- Loading and chunking PDF debate manuals
- Storing embeddings in ChromaDB
- Retrieving relevant judging criteria for evaluation

Will be implemented in Week 2.
"""


def load_and_index_documents(docs_dir: str) -> None:
    """
    Load PDF debate manuals, chunk them, embed, and store in ChromaDB.

    Args:
        docs_dir: Path to directory containing PDF debate manuals
    """
    # TODO: Implement in Week 2, Day 8-11
    raise NotImplementedError("RAG indexing not yet implemented")


def retrieve_judging_criteria(query: str, debate_format: str, top_k: int = 5) -> list:
    """
    Retrieve relevant judging criteria from ChromaDB for a given query.

    Args:
        query: The evaluation aspect (e.g., "rebuttal quality scoring")
        debate_format: The debate format (asian_parl, british_parl, wsdc)
        top_k: Number of relevant chunks to retrieve

    Returns:
        List of relevant text chunks from debate manuals
    """
    # TODO: Implement in Week 2, Day 12-13
    raise NotImplementedError("RAG retrieval not yet implemented")
