# DocuSense RAG evaluation

Mode: rerank; K=4; dataset SHA-256: `87b5a6e86843ce809ffe39036f884590243440372548c18abbd37cff39e1c326`

Automated measurements use author-assigned synthetic relevance labels. No manual judgments have been verified.
Lexical proxies and LLM judgments are not proof of factual correctness or entailment.
No-relevance cases are excluded from Recall/MRR and retained for abstention review.

| Metric | Value |
|---|---|
| recall_at_k | 1.0 |
| mrr_at_k | 1.0 |
| answerable_cases | 9 |
| retrieval_errors | 0 |
| answer_errors | 0 |
| judge_errors | 0 |
| answer_scored_cases | 0 |
| judge_scored_cases | 0 |
| rerank_fallbacks | 0 |
| expected_point_coverage_lexical | None |
| llm_judged_relevance | None |
| llm_judged_context_support | None |

## Comparison with vector baseline

Retrieval-only comparison; answer/judge averages require matching successful-case coverage. Check fallback and error counts before attributing differences to reranking.

- recall_at_k: +0.0000
- mrr_at_k: +0.0556

## Failure cases and examples

### revenue — retrieval checks passed

What was Orion Labs revenue in 2024?

Recall: 1.0; reciprocal rank: 1.0
Sources: finance-2024.pdf, finance-2022.pdf, finance-2024.pdf, procurement.pdf

Answer generation not requested.

### revenue-comparison — retrieval checks passed

Compare Orion Labs revenue in 2024 and 2023.

Recall: 1.0; reciprocal rank: 1.0
Sources: finance-2024.pdf, finance-2024.pdf, finance-2022.pdf, finance-2022.pdf

Answer generation not requested.

### current-expenses — retrieval checks passed

Under the current travel policy, how soon must I claim expenses?

Recall: 1.0; reciprocal rank: 1.0
Sources: travel-policy.pdf, travel-policy-archived.pdf, travel-policy.pdf, travel-policy-archived.pdf

Answer generation not requested.

### hotel — retrieval checks passed

What is the current domestic hotel cap, and are receipts required?

Recall: 1.0; reciprocal rank: 1.0
Sources: travel-policy.pdf, travel-policy.pdf, travel-policy-archived.pdf, procurement.pdf

Answer generation not requested.

### security — retrieval checks passed

How quickly must I report a suspected security incident, and what should I do with affected credentials?

Recall: 1.0; reciprocal rank: 1.0
Sources: security-policy.pdf, onboarding.pdf, support-sla.pdf, travel-policy.pdf

Answer generation not requested.

### sla — retrieval checks passed

Compare the response targets for Priority 1 and Priority 2 support.

Recall: 1.0; reciprocal rank: 1.0
Sources: support-sla.pdf, support-sla.pdf, disaster-recovery.pdf, security-policy.pdf

Answer generation not requested.

### new-hire — retrieval checks passed

When must a new hire finish security training?

Recall: 1.0; reciprocal rank: 1.0
Sources: onboarding.pdf, onboarding.pdf, security-policy.pdf, leave-policy.pdf

Answer generation not requested.

### termination — retrieval checks passed

How much notice is required to terminate the vendor contract?

Recall: 1.0; reciprocal rank: 1.0
Sources: vendor-contract.pdf, vendor-contract.pdf, procurement.pdf, procurement.pdf

Answer generation not requested.

### backup-vs-recovery — retrieval checks passed

What is backup retention and what are the recovery time and recovery point objectives?

Recall: 1.0; reciprocal rank: 1.0
Sources: disaster-recovery.pdf, security-policy.pdf, support-sla.pdf, disaster-recovery.pdf

Answer generation not requested.

### unknown — retrieval unscored

What is the CEO personal phone number?

Recall: None; reciprocal rank: None
Sources: support-sla.pdf, finance-2024.pdf, finance-2024.pdf, finance-2022.pdf

Answer generation not requested.

