import os
from pathlib import Path
from dotenv import load_dotenv
from filelock import FileLock
from redis import Redis

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

FAISS_INDEX_DIR = str(Path(__file__).parent / "faiss_store")
FAISS_LOCK_FILE = str(Path(__file__).parent / "faiss_store.lock")
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150

# ── RQ Job ─────────────────────────────────────────────────────────────────────
# This function is enqueued by streamlit_app.py and executed by the RQ worker.
# Flow: Load PDF → Chunk → Embed → Write to shared FAISS index on disk.
def ingest_document(file_path: str, file_hash: str, doc_name: str) -> None:
    """
    Background job: ingest a single PDF into the shared FAISS vector store.

    Args:
        file_path : absolute path to the saved PDF file (in uploads/)
        file_hash : MD5 hash of the file content (used as unique doc ID)
        doc_name  : original filename shown in the UI
    """
    r = Redis(decode_responses=True)
    r.set(f"doc:{file_hash}:status", "processing")

    try:
        # Step 1 – Load PDF pages
        loader = PyPDFLoader(file_path)
        pages  = loader.load()

        # Step 2 – Split into overlapping chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(pages)

        # Step 3 – Enrich each chunk with document metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["document_name"] = doc_name
            chunk.metadata["page_number"]   = chunk.metadata.get("page", 0)
            chunk.metadata["chunk_id"]      = f"{file_hash}_{i}"

        # Step 4 – Embed + update shared FAISS index (file-locked for safety)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        lock = FileLock(FAISS_LOCK_FILE, timeout=120)
        with lock:
            if os.path.exists(FAISS_INDEX_DIR):
                # Load existing index and add new document chunks
                index = FAISS.load_local(
                    FAISS_INDEX_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                index.add_documents(chunks)
            else:
                # First document — create the index from scratch
                index = FAISS.from_documents(chunks, embeddings)

            index.save_local(FAISS_INDEX_DIR)

        r.set(f"doc:{file_hash}:status", "completed")

    except Exception as e:
        r.set(f"doc:{file_hash}:status", "failed")
        r.set(f"doc:{file_hash}:error",  str(e))
        raise   # let RQ mark the job as failed too
