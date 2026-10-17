"""PDF chunks, Pinecone vector search, and optional hosted reranking."""
import hashlib
import math
from dataclasses import fields, is_dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from .config import CHUNK_SIZE, CHUNK_OVERLAP, DocuSenseError


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def as_dict(value):
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        # Pinecone SDK dataclasses can hold objects that cannot be deep-copied.
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return value.to_dict()


def pdf_chunks(path: Path, source: str, owner_id: str):
    try:
        if path.stat().st_size > 25 * 1024 * 1024:
            raise DocuSenseError('PDF exceeds the 25 MB ingestion limit.')
        content = path.read_bytes()
    except OSError:
        raise DocuSenseError('Cannot read the PDF file.') from None
    if b'%PDF-' not in content[:1024]:
        raise DocuSenseError('Invalid PDF file header.')
    document_id = digest(content)
    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception:
        raise DocuSenseError('Invalid, encrypted, or unreadable PDF.') from None
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise DocuSenseError('No extractable text found. Scanned PDFs require OCR before upload.')
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            'document_id': document_id, 'source': Path(source).name,
            'page': int(chunk.metadata.get('page', 0)), 'chunk_index': i,
            'owner_id': owner_id, 'chunk_id': f'{document_id}:{i:06d}',
        }
    return document_id, chunks


def search_vectors(store, query: str, active, *, top_k=4):
    """Search only manifest-published revisions, then verify returned metadata."""
    if not 1 <= top_k <= 100:
        raise DocuSenseError('Retrieval top_k must be between 1 and 100.')
    if not query.strip() or not active:
        return []
    filters = store._filter()
    filters['$or'] = [{'document_id': {'$eq': row['document_id']},
                       'revision': {'$eq': row['revision']}} for row in active]
    try:
        vector = store.embeddings.embed_query(query)
        store._validate_vector(vector)
    except DocuSenseError:
        raise
    except Exception:
        raise DocuSenseError('Query embedding failed. Please retry later.') from None
    try:
        result = as_dict(store.index.query(vector=vector, namespace=store.namespace,
            filter=filters, top_k=top_k, include_metadata=True))
    except Exception:
        raise DocuSenseError('Pinecone search failed. Please retry later.') from None
    allowed = {(row['document_id'], row['revision']) for row in active}
    documents = []
    for item in result.get('matches', []):
        match = as_dict(item)
        metadata = dict(match.get('metadata') or {})
        if metadata.get('owner_id') != store.owner_id or (metadata.get('document_id'), metadata.get('revision')) not in allowed:
            continue
        text = metadata.pop('text', '')
        if text:
            metadata['vector_score'] = match.get('score')
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


class RAGRetriever:
    def __init__(self, store, *, enabled=False, model='bge-reranker-v2-m3', candidates=20, top_k=4):
        if not 1 <= top_k <= candidates <= 100:
            raise DocuSenseError('Require 1 <= top_k <= candidates <= 100.')
        self.store, self.enabled, self.model = store, enabled, model
        self.candidates, self.top_k = candidates, top_k
        self.last_status = 'not_run'
        self._model_available = False

    def retrieve(self, query):
        # Retrieval errors propagate; never turn a provider failure into a fake success.
        docs = self.store.retrieve(query, top_k=self.candidates if self.enabled else self.top_k)
        return self.rank(query, docs)

    def rank(self, query, candidates):
        self.last_status = 'vector'
        fallback = candidates[:self.top_k]
        if not candidates:
            self.last_status = 'empty'
            return []
        if not self.enabled:
            return fallback
        try:
            if not self._model_available:
                available = self.store.client.inference.list_models(type='rerank')
                names = {getattr(m, 'model', None) or as_dict(m).get('model') for m in available}
                if self.model not in names:
                    self.last_status = 'fallback:model_unavailable'
                    return fallback
                self._model_available = True
            response = self.store.client.inference.rerank(
                model=self.model, query=query,
                documents=[{'id':d.metadata['chunk_id'], 'text':d.page_content} for d in candidates],
                rank_fields=['text'], top_n=min(self.top_k, len(candidates)), return_documents=False)
            data = as_dict(response)['data']
            if len(data) != min(self.top_k, len(candidates)):
                raise ValueError('Incomplete ranking')
            ranked, seen = [], set()
            for raw in data:
                row = as_dict(raw)
                i, score = row['index'], row['score']
                if type(i) is not int or i < 0 or i >= len(candidates) or i in seen or not math.isfinite(score):
                    raise ValueError('Invalid ranking')
                seen.add(i)
                original = candidates[i]
                # Reuse original context and metadata, never trust returned document payloads.
                ranked.append(Document(page_content=original.page_content,
                                       metadata=dict(original.metadata, rerank_score=score)))
            self.last_status = 'reranked'
            return ranked
        except Exception:
            self.last_status = 'fallback:api_or_response_error'
            return fallback


def configured_retriever(store, settings):
    return RAGRetriever(store, enabled=settings.rerank_enabled, model=settings.rerank_model,
                        candidates=settings.candidate_k, top_k=settings.top_k)
