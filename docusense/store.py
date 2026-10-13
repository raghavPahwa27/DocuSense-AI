"""Bring-your-own Gemini vectors. No FAISS loading, pickle, or integrated embeddings."""
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, fields, is_dataclass
from contextlib import contextmanager, closing
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from .config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DocuSenseError


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def as_dict(value):
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        # SDK dataclasses can contain OpenAPI objects that cannot be deep-copied.
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


class Manifest:
    """Local publication ledger: unverified/partial vectors are never searchable.

    Each method uses a short SQLite transaction. Keep this directory durable.
    This intentionally supports one local application deployment, not a cluster.
    """
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS documents (owner TEXT, document_id TEXT, '
                       'revision TEXT, source TEXT, chunk_count INTEGER, status TEXT, '
                       'PRIMARY KEY(owner, document_id))')
        path.chmod(0o600)

    @contextmanager
    def connect(self):
        try:
            with closing(sqlite3.connect(self.path, timeout=30)) as db:
                with db:
                    yield db
        except sqlite3.Error:
            raise DocuSenseError('Local document state is unavailable. Check storage permissions and retry.') from None

    def documents(self, owner, active_only=True):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('SELECT * FROM documents WHERE owner=?' +
                              (" AND status='active'" if active_only else ''), (owner,)).fetchall()
        return [dict(row) for row in rows]

    def publish(self, owner, doc_id, revision, source, count):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)',
                       (owner, doc_id, revision, source, count, 'active'))

    def hide(self, owner, doc_id):
        with self.connect() as db:
            db.execute("UPDATE documents SET status='delete_pending' WHERE owner=? AND document_id=?",
                       (owner, doc_id))

    def remove(self, owner, doc_id):
        with self.connect() as db:
            db.execute('DELETE FROM documents WHERE owner=? AND document_id=?', (owner, doc_id))


@dataclass
class IngestionResult:
    document_id: str
    chunk_count: int
    duplicate: bool


class PineconeStore:
    def __init__(self, client, embeddings, settings, owner_id: str, *, verify_attempts=8):
        if not owner_id or not owner_id.strip():
            raise DocuSenseError('A nonempty owner/session ID is required.')
        self.client, self.embeddings, self.settings = client, embeddings, settings
        self.owner_id = owner_id
        self.namespace = 'docusense-' + digest(owner_id.encode())
        self.verify_attempts = verify_attempts
        # Probe actual output; do not assume a model's default dimensionality.
        try:
            doc = embeddings.embed_documents(['DocuSense dimension verification.'])[0]
            query = embeddings.embed_query('DocuSense dimension verification.')
            self.dimension = len(doc)
            self._validate_vector(doc)
            self._validate_vector(query)
        except DocuSenseError:
            raise
        except Exception:
            raise DocuSenseError('Gemini embedding probe failed. Check credentials and service availability.') from None
        try:
            description = as_dict(client.describe_index(settings.index_name))
        except Exception:
            raise DocuSenseError('Cannot describe the Pinecone index. Check its name, credentials, and connectivity.') from None
        self._validate_index(description)
        try:
            self.index = client.Index(host=description['host'])
        except Exception:
            raise DocuSenseError('Cannot connect to the configured Pinecone index.') from None
        # A different remote index must never reuse the local publication ledger.
        ledger = digest(description['host'].encode())[:24] + '.sqlite3'
        self.manifest = Manifest(settings.state_dir / ledger)

    def _validate_vector(self, vector):
        if len(vector) != self.dimension or not vector or not all(math.isfinite(x) for x in vector) or not any(vector):
            raise DocuSenseError('Gemini returned an invalid or incompatible embedding vector.')

    def _validate_index(self, description):
        if description.get('dimension') != self.dimension:
            raise DocuSenseError(f'Pinecone dimension must match the measured Gemini dimension ({self.dimension}).')
        if description.get('metric') != 'cosine' or description.get('vector_type', 'dense') != 'dense':
            raise DocuSenseError('Pinecone must use dense vectors and the cosine metric.')
        if description.get('embed'):
            raise DocuSenseError('Integrated embedding indexes are unsupported; use a standard vector index.')
        if not description.get('status', {}).get('ready'):
            raise DocuSenseError('Pinecone index is not ready. Retry after initialization.')
        # Pod selective metadata indexing must include every field used by filters.
        spec = description.get('spec', {})
        indexed = spec.get('pod', {}).get('metadata_config', {}).get('indexed')
        if indexed is not None and not {'owner_id', 'document_id', 'revision'}.issubset(indexed):
            raise DocuSenseError('Index metadata filtering must include owner_id, document_id, and revision.')
        schema = spec.get('serverless', {}).get('schema', {}).get('fields')
        if schema is not None and not all(schema.get(k, {}).get('filterable') for k in ('owner_id', 'document_id', 'revision')):
            raise DocuSenseError('Index schema must allow owner_id, document_id, and revision filters.')

    def _filter(self, document_id=None, revision=None):
        result = {'owner_id': {'$eq': self.owner_id}}
        if document_id:
            result['document_id'] = {'$eq': document_id}
        if revision:
            result['revision'] = {'$eq': revision}
        return result

    def ingest_pdf(self, path: Path, source: Optional[str] = None, *, reindex=False):
        doc_id, chunks = pdf_chunks(Path(path), source or Path(path).name, self.owner_id)
        return self.ingest_chunks(doc_id, chunks, reindex=reindex)

    def ingest_chunks(self, document_id, chunks, *, reindex=False):
        if not chunks:
            raise DocuSenseError('The document has no text chunks.')
        existing = next((r for r in self.manifest.documents(self.owner_id, False)
                         if r['document_id'] == document_id), None)
        if existing and existing['status'] == 'delete_pending':
            raise DocuSenseError('Finish the pending deletion before ingesting this document again.')
        signature = json.dumps([[c.page_content, c.metadata] for c in chunks], sort_keys=True)
        revision = digest((EMBEDDING_MODEL + str(self.dimension) + signature).encode())
        if existing and not reindex:
            return IngestionResult(document_id, existing['chunk_count'], True)
        records = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            if metadata.get('owner_id') != self.owner_id or metadata.get('document_id') != document_id:
                raise DocuSenseError('Chunk ownership does not match this document session.')
            if not all(k in metadata for k in ('source', 'page', 'chunk_index', 'chunk_id')):
                raise DocuSenseError('Chunk citation metadata is incomplete.')
            metadata.update(text=chunk.page_content, revision=revision)
            records.append({'id': metadata['chunk_id'] + ':' + revision, 'metadata': metadata})
        try:
            # Embed before overwriting any vectors. Batches stay below provider limits.
            for start in range(0, len(chunks), 32):
                vectors = self.embeddings.embed_documents([c.page_content for c in chunks[start:start + 32]])
                if len(vectors) != len(chunks[start:start + 32]):
                    raise DocuSenseError('Gemini returned an incomplete embedding batch.')
                for record, vector in zip(records[start:start + 32], vectors):
                    self._validate_vector(vector)
                    record['values'] = vector
        except DocuSenseError:
            raise
        except Exception:
            raise DocuSenseError('Document embedding failed. Existing documents remain available.') from None
        try:
            for start in range(0, len(records), 32):
                self.index.upsert(vectors=records[start:start + 32], namespace=self.namespace)
            self._verify(records, document_id, revision)
        except DocuSenseError:
            raise
        except Exception:
            raise DocuSenseError('Pinecone ingestion failed; retry safely. Unpublished vectors are excluded from search.') from None
        self.manifest.publish(self.owner_id, document_id, revision, chunks[0].metadata['source'], len(chunks))
        # Previous revisions remain hidden; document deletion removes all revisions.
        return IngestionResult(document_id, len(chunks), False)

    def _verify(self, records, document_id, revision):
        for attempt in range(self.verify_attempts):
            complete = True
            for start in range(0, len(records), 100):
                batch = records[start:start + 100]
                found = as_dict(self.index.fetch(ids=[r['id'] for r in batch], namespace=self.namespace)).get('vectors', {})
                complete = complete and all(
                    r['id'] in found and as_dict(found[r['id']]).get('metadata') == r['metadata'] for r in batch)
            # Verify query visibility too; a successful upsert alone is insufficient.
            response = as_dict(self.index.query(vector=records[0]['values'], top_k=1,
                namespace=self.namespace, filter=self._filter(document_id, revision), include_metadata=True))
            matches = response.get('matches', [])
            if complete and matches and as_dict(matches[0]).get('id') in {r['id'] for r in records}:
                return
            if attempt + 1 < self.verify_attempts:
                time.sleep(min(2 ** attempt, 5))
        raise DocuSenseError('Pinecone verification timed out. Retry ingestion; this revision is not published.')

    def retrieve(self, query: str, *, top_k=4, document_id=None):
        if not 1 <= top_k <= 100:
            raise DocuSenseError('Retrieval top_k must be between 1 and 100.')
        active = self.manifest.documents(self.owner_id)
        if document_id:
            active = [r for r in active if r['document_id'] == document_id]
        if not query.strip() or not active:
            return []
        filters = self._filter()
        filters['$or'] = [{'document_id': {'$eq': r['document_id']}, 'revision': {'$eq': r['revision']}} for r in active]
        try:
            vector = self.embeddings.embed_query(query)
            self._validate_vector(vector)
        except DocuSenseError:
            raise
        except Exception:
            raise DocuSenseError('Query embedding failed. Please retry later.') from None
        try:
            result = as_dict(self.index.query(vector=vector, namespace=self.namespace,
                filter=filters, top_k=top_k, include_metadata=True))
        except Exception:
            raise DocuSenseError('Pinecone search failed. Please retry later.') from None
        allowed = {(r['document_id'], r['revision']) for r in active}
        documents = []
        for item in result.get('matches', []):
            match = as_dict(item)
            metadata = dict(match.get('metadata') or {})
            # Defense in depth: check provider response as well as query filters.
            if metadata.get('owner_id') != self.owner_id or (metadata.get('document_id'), metadata.get('revision')) not in allowed:
                continue
            text = metadata.pop('text', '')
            if text:
                metadata['vector_score'] = match.get('score')
                documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def delete_document(self, document_id):
        if not any(r['document_id'] == document_id for r in self.manifest.documents(self.owner_id, False)):
            raise DocuSenseError('Document is not registered to this session.')
        self.manifest.hide(self.owner_id, document_id)
        try:
            self.index.delete(namespace=self.namespace, filter=self._filter(document_id))
        except Exception:
            raise DocuSenseError('Remote deletion failed. Document is hidden locally; retry deletion to finish.') from None
        self.manifest.remove(self.owner_id, document_id)


def connect_store(settings, owner_id):
    from functools import partial
    from pinecone import Pinecone
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=settings.google_api_key)
    # Bound provider retries/timeouts without changing model/task/dimension settings.
    embeddings.client.batch_embed_contents = partial(embeddings.client.batch_embed_contents, timeout=30, retry=None)
    return PineconeStore(Pinecone(api_key=settings.pinecone_api_key), embeddings, settings, owner_id)
