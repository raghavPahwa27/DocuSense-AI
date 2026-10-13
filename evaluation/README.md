# Reproducible RAG evaluation

`dataset.json` is a version-controlled, synthetic corpus (12 two-page documents, 24 chunks) with 10 questions. Labels specify exact chunk IDs, source names and expected answer points, including archived-policy distractors and a question the corpus cannot answer. These are author-assigned labels, not independently verified human ground truth. No private uploaded documents are included.

```bash
venv/bin/python -m evaluation.run --ingest --answers --judge
```

This uses the same Gemini embedding and Pinecone storage code as the application, independent of Streamlit. It writes into a dataset-hash-specific evaluation owner/namespace. Ingestion is idempotent. The source dataset hash, configuration, per-case retrieved context, answers and failures are recorded in JSON, with a readable Markdown report alongside it. Reports contain document text: keep reports for private custom corpora out of version control. Only checked synthetic reports belong in `evaluation/reports/`.

Measurements:

- Recall@K = number of unique relevant chunk IDs in the first K results divided by all expected relevant IDs.
- MRR@K = average reciprocal rank of the first relevant result, truncated at K. Queries without an expected relevant chunk are excluded from retrieval averages and retained for abstention review.
- Retrieval failures count as zero for answerable questions, rather than disappearing from averages.
- Expected-point coverage checks normalized aliases. Token overlap is a weak lexical context-support proxy. Neither establishes semantic relevance or entailment.
- Optional configured-chat-model judgments score answer relevance and support from context. They are automated judgments from the same model family, vulnerable to bias and errors; they are not human verification. API/judge failures receive no invented answer scores and are counted separately.
- Every report case contains `manual_judgment: null` until a human actually reviews it. Review the question, retrieved context, answer, citations and expected points; retain reviewer identity, date and rationale in a separate reviewed artifact.

The small synthetic corpus is a smoke benchmark, not evidence of enterprise quality. Expand it with approved domain questions, negative cases, difficult distractors, real pagination and multiple annotators before tuning on a separate validation split. Do not choose settings based only on the same small dataset used for development.

## Reranking comparison

Establish and retain a vector baseline first, then use the **same dataset and K**:

```bash
venv/bin/python -m evaluation.run --ingest --output evaluation/reports/vector-baseline.json
venv/bin/python -m evaluation.run --mode rerank --candidates 20 --compare-with evaluation/reports/vector-baseline.json --output evaluation/reports/rerank-20.json
venv/bin/python -m evaluation.run --mode rerank --candidates 16 --compare-with evaluation/reports/vector-baseline.json --output evaluation/reports/rerank-16.json
```

For answer evaluation add `--answers --judge --llm-interval 15`. The interval applies between answer and judge requests; use a longer interval if your account quota requires it. The configured chat provider/model is used for both answer generation and automated judgments. API failures count separately and the process returns a nonzero exit code; inspect score coverage before comparing answer averages. This measures direct context-based answer synthesis, not every decision path through the interactive ReAct agent.

Use `--rerank-model MODEL` to evaluate another model returned by `pc.inference.list_models(type='rerank')`. `--mode vector` explicitly bypasses reranking regardless of `.env`. Candidate counts must be between K and 100. Query vectors come from Gemini; reranking calls `pc.inference.rerank()` after a normal `index.query()`.

Fallback counts and each query's actual retrieval mode are recorded. A rerank report with fallback results is not evidence that the hosted reranker succeeded. Configuration comparisons reject mismatched dataset hashes, K, embedding model/dimension and corpus size. Identical metric scores on this small corpus do not establish equivalent quality in the real application.

## Recorded run limitations

The initial pre-rerank baseline has 3 successful answers/judgments and 7 answer failures from the observed per-minute quota. The paced repeat has 7 successful answers, 6 judgments, 3 answer failures and 1 judge failure from the observed **20 requests/day** quota. Future runs stop further answer/judge API requests after a quota exception while still collecting retrieval metrics. Reset/raise your account quota before repeating a full answer comparison. The checked-in rerank comparisons measure retrieval only; they do not establish improved answer quality.
