"""Deterministic retrieval metrics and clearly labeled lexical answer proxies."""
import re


def recall_at_k(retrieved, relevant, k):
    if k < 1:
        raise ValueError('k must be positive')
    relevant = set(relevant)
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else None


def reciprocal_rank(retrieved, relevant, k=None):
    relevant = set(relevant)
    if not relevant:
        return None
    return next((1.0 / i for i, key in enumerate(retrieved[:k], 1) if key in relevant), 0.0)


def mean(values):
    measured = [v for v in values if v is not None]
    return sum(measured) / len(measured) if measured else None


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


def answer_proxies(answer, expected_points, contexts):
    """These are lexical checks, NOT correctness or semantic entailment judgments."""
    normalized = normalize(answer)
    hits = [any(normalize(alias) in normalized for alias in point['aliases']) for point in expected_points]
    answer_tokens = set(normalized.split())
    context_tokens = set(normalize(' '.join(contexts)).split())
    return {
        'expected_point_coverage_lexical': sum(hits) / len(hits) if hits else None,
        'answer_context_token_overlap': len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else None,
        'missing_expected_points': [p['point'] for p, hit in zip(expected_points, hits) if not hit],
    }
