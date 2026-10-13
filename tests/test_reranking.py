from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from conftest import chunk
from docusense.retrieval import RAGRetriever


def retriever(rows=None, names=None):
    docs = [chunk(index=i, text=f'Context {i}') for i in range(6)]
    inference = SimpleNamespace(
        list_models=Mock(return_value=[{'model':m} for m in (names if names is not None else ['bge-reranker-v2-m3'])]),
        rerank=Mock(return_value={'data':rows if rows is not None else [dict(index=i,score=0.9) for i in [4,2,1,0]]}))
    store = SimpleNamespace(client=SimpleNamespace(inference=inference), retrieve=Mock(return_value=docs))
    return RAGRetriever(store, enabled=True), docs, inference


def test_rerank_reorders_preserving_metadata():
    r, docs, inference = retriever()
    result = r.retrieve('question')
    assert [d.metadata['chunk_index'] for d in result] == [4,2,1,0]
    assert result[0].page_content == docs[4].page_content
    for key in ['owner_id','document_id','source','page','chunk_id']:
        assert result[0].metadata[key] == docs[4].metadata[key]
    assert 'rerank_score' not in docs[4].metadata
    r.store.retrieve.assert_called_once_with('question', top_k=20)
    assert r.last_status == 'reranked'


@pytest.mark.parametrize('rows', [[], [{'index':-1,'score':0.9}]*4, [{'index':1,'score':0.9}]*4, [{'index':999,'score':0.9}]*4])
def test_bad_rerank_responses_fall_back(rows):
    r, docs, _ = retriever(rows=rows)
    assert r.retrieve('q') == docs[:4]
    assert r.last_status.startswith('fallback:')


def test_missing_model_falls_back():
    r, docs, inference = retriever(names=[])
    assert r.retrieve('q') == docs[:4]
    inference.rerank.assert_not_called()
    assert r.last_status == 'fallback:model_unavailable'


def test_api_failure_falls_back():
    r, docs, inference = retriever()
    inference.rerank.side_effect = RuntimeError('private error')
    assert r.retrieve('q') == docs[:4]
    assert r.last_status.startswith('fallback:')


def test_disabled_and_empty_do_not_call_inference():
    r, docs, inference = retriever()
    r.enabled = False
    assert r.retrieve('q') == docs[:4]
    r.enabled = True
    r.store.retrieve.return_value = []
    assert r.retrieve('q') == []
    inference.rerank.assert_not_called()
    inference.list_models.assert_not_called()
