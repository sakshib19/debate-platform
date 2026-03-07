"""
RAG module: Retrieval-Augmented Generation over debate manuals.

Handles:
- Loading and chunking PDF debate manuals
- Storing embeddings in ChromaDB (persistent, disk-backed)
- Retrieving relevant judging criteria for speaker evaluation

Flow:
  1. load_and_index_documents() — one-time: reads PDFs → chunks → embeds → stores
  2. retrieve_judging_criteria() — per-evaluation: similarity search over stored chunks
"""

import os
import logging
from typing import List

from app.config import settings

logger = logging.getLogger(__name__)

# Persistent ChromaDB directory (next to backend/)
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

# Singleton Chroma client + collection
_collection = None


def _get_collection():
    """Return (or create) the ChromaDB collection, reusing a single client."""
    global _collection
    if _collection is None:
        import chromadb

        os.makedirs(CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Uses Chroma's built-in ONNX all-MiniLM-L6-v2 embeddings (no torch needed)
        _collection = client.get_or_create_collection(
            name="debate_manuals",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection ready (%d docs)", _collection.count())
    return _collection


def load_and_index_documents(docs_dir: str) -> dict:
    """
    Load PDF debate manuals from *docs_dir*, chunk them, and store in ChromaDB.

    Skips re-indexing if the collection already contains documents.
    Call with force=True via the /index-documents endpoint to re-index.

    Returns:
        {"indexed": <number of chunks stored>, "skipped": <files ignored>}
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    collection = _get_collection()

    # Gather PDF paths
    pdf_paths = sorted(
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    logger.info("Found %d PDFs in %s", len(pdf_paths), docs_dir)

    # Load pages from all PDFs
    all_pages = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = os.path.basename(path)
        all_pages.extend(pages)
        logger.info("  Loaded %s (%d pages)", os.path.basename(path), len(pages))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_pages)
    logger.info("Split into %d chunks", len(chunks))

    # Prepare for ChromaDB (no external embedding model — using Chroma's default)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk.metadata.get("source_file", ""),
            "page": chunk.metadata.get("page", 0),
        }
        for chunk in chunks
    ]

    # Upsert (idempotent — safe to call repeatedly)
    BATCH = 500
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info("Indexed %d chunks into ChromaDB", len(chunks))
    return {"indexed": len(chunks), "pdf_files": len(pdf_paths)}


def retrieve_judging_criteria(query: str, debate_format: str = "", top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant judging-criteria chunks for a query.

    Args:
        query:         Natural-language question (e.g. "how to score rebuttal quality")
        debate_format: Optional filter hint (currently unused — small corpus)
        top_k:         Number of chunks to return

    Returns:
        List of plain-text chunks, most relevant first.
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — call load_and_index_documents first")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    # results["documents"] is List[List[str]]; flatten the outer list
    chunks = results["documents"][0] if results["documents"] else []
    logger.info("Retrieved %d chunks for query: %.60s…", len(chunks), query)
    return chunks
