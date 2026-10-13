import pytest
from evaluation.metrics import recall_at_k, reciprocal_rank, answer_proxies
from evaluation.run import load_dataset, corpus_chunks, evaluate


def test_known_retrieval_metrics():
    assert recall_at_k(['x','a','a','b'], ['a','b'], 3) == 0.5
    assert reciprocal_rank(['x','a','b'], ['a','b'], 3) == 0.5
    assert recall_at_k([], ['a'], 4) == 0
    assert reciprocal_rank(['x'], ['a'], 4) == 0
    assert recall_at_k(['x'], [], 4) is None
    assert reciprocal_rank([], [], 4) is None
    with pytest.raises(ValueError): recall_at_k([], ['a'], 0)


def test_dataset_labels_match_actual_chunks():
    dataset, fingerprint = load_dataset()
    chunks = [c for _,docs in corpus_chunks(dataset, 'evaluation') for c in docs]
    assert len(chunks) == 24
    ids = {c.metadata['chunk_id'] for c in chunks}
    assert all(set(c['relevant_chunk_ids']) <= ids for c in dataset['cases'])
    assert len({c['id'] for c in dataset['cases']}) == len(dataset['cases'])


def test_answer_proxies_are_explicitly_lexical():
    result = answer_proxies('The cap is 180.', [{'point':'180 dollars','aliases':['180']}], ['The cap is 180.'])
    assert result['expected_point_coverage_lexical'] == 1
    assert result['answer_context_token_overlap'] == 1


def test_retrieval_failures_not_dropped_from_metrics():
    dataset, _ = load_dataset()
    def fail(q): raise RuntimeError('secret')
    report = evaluate(dataset, fail)
    assert report['metrics']['recall_at_k'] == 0
    assert report['metrics']['mrr_at_k'] == 0
    assert report['metrics']['retrieval_errors'] == 10
    assert all(r['manual_judgment'] is None for r in report['cases'])
    assert 'secret' not in str(report)


def test_quota_failure_stops_further_llm_requests():
    from unittest.mock import Mock
    from google.api_core.exceptions import ResourceExhausted
    from evaluation.run import PacedLLM
    from docusense.config import DocuSenseError
    raw = Mock()
    raw.invoke.side_effect = ResourceExhausted('quota')
    llm = PacedLLM(raw, 0)
    with pytest.raises(ResourceExhausted): llm.invoke('question')
    with pytest.raises(DocuSenseError): llm.invoke('next question')
    assert raw.invoke.call_count == 1
