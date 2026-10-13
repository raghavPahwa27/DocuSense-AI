import pytest
from docusense.config import DocuSenseError
from docusense.store import PineconeStore, pdf_chunks
from conftest import chunk, FakeEmbeddings


def test_ingest_duplicate_retrieve_and_delete(setup_store):
    factory, client = setup_store
    store = factory()
    result = store.ingest_chunks('doc-a', [chunk()])
    assert not result.duplicate
    assert store.ingest_chunks('doc-a', [chunk()]).duplicate
    docs = store.retrieve('revenue')
    assert docs[0].metadata['page'] == 0
    assert docs[0].metadata['chunk_id'] == 'doc-a:000000'
    store.delete_document('doc-a')
    assert store.retrieve('revenue') == []
    assert not client.index.records[store.namespace]


def test_isolation_in_all_operations(setup_store):
    factory, client = setup_store
    alice, bob = factory(), factory('bob')
    alice.ingest_chunks('doc-a', [chunk()])
    bob.ingest_chunks('doc-a', [chunk(owner='bob', text='Secret Bob text')])
    assert alice.namespace != bob.namespace
    assert alice.retrieve('text')[0].page_content != 'Secret Bob text'
    alice.ingest_chunks('doc-a', [chunk(text='Updated')], reindex=True)
    assert alice.retrieve('text')[0].page_content == 'Updated'
    alice.delete_document('doc-a')
    assert bob.retrieve('text')[0].page_content == 'Secret Bob text'
    for op in client.index.calls:
        if op[0] in {'query', 'delete'}:
            assert 'owner_id' in op[2]


def test_partial_upload_not_published(setup_store, monkeypatch):
    factory, client = setup_store
    store = factory()
    monkeypatch.setattr(client.index, 'fetch', lambda **kwargs: {'vectors': {}})
    with pytest.raises(DocuSenseError, match='verification timed out'):
        store.ingest_chunks('doc-a', [chunk()])
    assert store.retrieve('revenue') == []


def test_failed_reindex_preserves_old_revision(setup_store, monkeypatch):
    factory, client = setup_store
    store = factory()
    store.ingest_chunks('doc-a', [chunk()])
    monkeypatch.setattr(client.index, 'upsert', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('secret body')))
    with pytest.raises(DocuSenseError, match='ingestion failed'):
        store.ingest_chunks('doc-a', [chunk(text='New')], reindex=True)
    assert store.retrieve('revenue')[0].page_content == 'Revenue was 4500000.'


@pytest.mark.parametrize('field,value', [('dimension',4), ('metric','dotproduct'), ('vector_type','sparse'), ('embed',{'model':'hosted'}), ('status',{'ready':False})])
def test_rejects_incompatible_index(setup_store, field, value):
    factory, client = setup_store
    client.description[field] = value
    with pytest.raises(DocuSenseError):
        factory()
    assert not client.index.calls


def test_sdk_fetch_response_dataclass():
    from pinecone import FetchResponse
    from docusense.store import as_dict
    assert as_dict(FetchResponse(namespace='test', vectors={}, usage={}))['vectors'] == {}


def test_sdk_fetch_does_not_deepcopy_usage():
    from pinecone import FetchResponse, Vector
    from docusense.store import as_dict
    class Usage:
        def __deepcopy__(self, memo):
            raise TypeError('SDK object is not copyable')
    r = FetchResponse(namespace='test', vectors={'x': Vector(id='x', values=[1.0], metadata={})}, usage=Usage())
    assert as_dict(as_dict(r)['vectors']['x'])['id'] == 'x'


def test_duplicate_content_renamed_is_noop(setup_store):
    factory, client = setup_store
    s = factory()
    s.ingest_chunks('doc-a', [chunk()])
    renamed = chunk()
    renamed.metadata['source'] = 'renamed.pdf'
    assert s.ingest_chunks('doc-a', [renamed]).duplicate
    assert len([c for c in client.index.calls if c[0] == 'upsert']) == 1


def test_delete_failure_hides_document_and_can_retry(setup_store, monkeypatch):
    factory, client = setup_store
    s = factory()
    s.ingest_chunks('doc-a', [chunk()])
    original = client.index.delete
    def fail(**kw):
        raise RuntimeError('sensitive provider response')
    monkeypatch.setattr(client.index, 'delete', fail)
    with pytest.raises(DocuSenseError, match='hidden locally'):
        s.delete_document('doc-a')
    assert s.retrieve('test') == []
    with pytest.raises(DocuSenseError, match='pending deletion'):
        s.ingest_chunks('doc-a', [chunk()])
    monkeypatch.setattr(client.index, 'delete', original)
    s.delete_document('doc-a')
    assert not s.manifest.documents(s.owner_id, False)


def test_foreign_document_delete_rejected(setup_store):
    factory, client = setup_store
    with pytest.raises(DocuSenseError, match='not registered'):
        factory('bob').delete_document('doc-a')
    assert not client.index.calls


def test_response_defense_in_depth(setup_store, monkeypatch):
    factory, client = setup_store
    s = factory()
    s.ingest_chunks('doc-a', [chunk()])
    foreign = chunk(owner='bob')
    monkeypatch.setattr(client.index, 'query', lambda **kw: {'matches': [
        {'metadata': dict(foreign.metadata, text='secret', revision='other')}]})
    assert s.retrieve('revenue') == []


def test_empty_and_invalid_pdfs(tmp_path):
    from pypdf import PdfWriter
    path = tmp_path / 'broken.pdf'
    path.write_bytes(b'not a pdf')
    with pytest.raises(DocuSenseError, match='Invalid PDF'):
        pdf_chunks(path, 'broken.pdf', 'alice')
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(str(path))
    with pytest.raises(DocuSenseError, match='No extractable text'):
        pdf_chunks(path, 'empty.pdf', 'alice')


def test_embedding_failure_is_redacted(setup_store, monkeypatch):
    factory, client = setup_store
    s = factory()
    def fail(texts):
        raise RuntimeError('private text and key')
    monkeypatch.setattr(s.embeddings, 'embed_documents', fail)
    with pytest.raises(DocuSenseError) as info:
        s.ingest_chunks('doc-a', [chunk()])
    assert 'private' not in str(info.value)
    assert not client.index.calls


def test_invalid_embedding_shape_rejected(setup_store, monkeypatch):
    factory, client = setup_store
    s = factory()
    monkeypatch.setattr(s.embeddings, 'embed_documents', lambda texts: [[1.0]])
    with pytest.raises(DocuSenseError, match='incompatible'):
        s.ingest_chunks('doc-a', [chunk()])


def test_query_failure_redacted(setup_store, monkeypatch):
    factory, client = setup_store
    s = factory()
    s.ingest_chunks('doc-a', [chunk()])
    def fail(**kw):
        raise RuntimeError('private text')
    monkeypatch.setattr(client.index, 'query', fail)
    with pytest.raises(DocuSenseError, match='Pinecone search failed'):
        s.retrieve('revenue')
