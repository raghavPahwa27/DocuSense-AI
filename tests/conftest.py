from copy import deepcopy
from types import SimpleNamespace
import pytest
from docusense.config import Settings
from docusense.store import PineconeStore
from langchain_core.documents import Document


def matches(metadata, filters):
    return all(any(matches(metadata, f) for f in value) if key == '$or'
               else metadata.get(key) == value['$eq'] for key, value in filters.items())

class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, float(len(t) % 7 + 1), 0.5] for t in texts]
    def embed_query(self, text):
        return self.embed_documents([text])[0]

class FakeIndex:
    def __init__(self):
        self.records = {}
        self.calls = []
    def upsert(self, vectors, namespace):
        self.calls.append(('upsert', namespace))
        self.records.setdefault(namespace, {}).update({r['id']: deepcopy(r) for r in vectors})
    def fetch(self, ids, namespace):
        return {'vectors': {i: self.records.get(namespace, {})[i] for i in ids if i in self.records.get(namespace, {})}}
    def query(self, vector, top_k, namespace, filter, include_metadata):
        self.calls.append(('query', namespace, deepcopy(filter)))
        rows = [dict(r, score=1.0) for r in self.records.get(namespace, {}).values() if matches(r['metadata'], filter)]
        return {'matches': rows[:top_k]}
    def delete(self, namespace, filter):
        self.calls.append(('delete', namespace, deepcopy(filter)))
        self.records[namespace] = {i:r for i,r in self.records.get(namespace, {}).items() if not matches(r['metadata'], filter)}

class FakeClient:
    def __init__(self):
        self.index = FakeIndex()
        self.description = {'dimension': 3, 'metric': 'cosine', 'vector_type': 'dense', 'host': 'test-index', 'status': {'ready': True}}
    def describe_index(self, name):
        return self.description
    def Index(self, host):
        return self.index

@pytest.fixture
def setup_store(tmp_path):
    client = FakeClient()
    settings = Settings('private-google-key', 'private-pinecone-key', 'test', tmp_path)
    def factory(owner='alice'):
        return PineconeStore(client, FakeEmbeddings(), settings, owner, verify_attempts=1)
    return factory, client


def chunk(owner='alice', doc='doc-a', index=0, text='Revenue was 4500000.'):
    return Document(page_content=text, metadata=dict(owner_id=owner, document_id=doc,
        source='report.pdf', page=0, chunk_index=index, chunk_id=f'{doc}:{index:06d}'))
