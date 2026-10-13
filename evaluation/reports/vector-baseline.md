# DocuSense RAG evaluation

Mode: vector; K=4; dataset SHA-256: `87b5a6e86843ce809ffe39036f884590243440372548c18abbd37cff39e1c326`

Automated measurements use author-assigned synthetic relevance labels. No manual judgments have been verified.
Lexical proxies and LLM judgments are not proof of factual correctness or entailment.
No-relevance cases are excluded from Recall/MRR and retained for abstention review.

| Metric | Value |
|---|---|
| recall_at_k | 1.0 |
| mrr_at_k | 0.9444444444444444 |
| answerable_cases | 9 |
| retrieval_errors | 0 |
| answer_errors | 7 |
| judge_errors | 0 |
| expected_point_coverage_lexical | 1.0 |
| llm_judged_relevance | 1.0 |
| llm_judged_context_support | 1.0 |
| answer_scored_cases | 3 |
| judge_scored_cases | 3 |
| rerank_fallbacks | 0 |

## Failure cases and examples

### revenue — retrieval checks passed

What was Orion Labs revenue in 2024?

Recall: 1.0; reciprocal rank: 1.0
Sources: finance-2024.pdf, finance-2024.pdf, finance-2022.pdf, vendor-contract.pdf

Orion Labs' total revenue in 2024 was $4,500,000 [finance-2024.pdf | page 1].

### revenue-comparison — retrieval checks passed

Compare Orion Labs revenue in 2024 and 2023.

Recall: 1.0; reciprocal rank: 1.0
Sources: finance-2024.pdf, finance-2024.pdf, finance-2022.pdf, vendor-contract.pdf

In 2024, Orion Labs' total revenue was $4,500,000 [finance-2024.pdf | page 1], compared to $3,800,000 in 2023 [finance-2024.pdf | page 2]. This represents an increase of $700,000 from 2023 to 2024.

### current-expenses — retrieval checks passed

Under the current travel policy, how soon must I claim expenses?

Recall: 1.0; reciprocal rank: 0.5
Sources: travel-policy-archived.pdf, travel-policy.pdf, travel-policy.pdf, travel-policy-archived.pdf

Under the current travel policy, employees must submit expense claims within 30 days of completing travel [travel-policy.pdf | page 1].

### hotel — review needed

What is the current domestic hotel cap, and are receipts required?

Recall: 1.0; reciprocal rank: 1.0
Sources: travel-policy.pdf, travel-policy-archived.pdf, travel-policy.pdf, travel-policy-archived.pdf

Answer generation failed; no answer score assigned.

### security — review needed

How quickly must I report a suspected security incident, and what should I do with affected credentials?

Recall: 1.0; reciprocal rank: 1.0
Sources: security-policy.pdf, onboarding.pdf, security-policy.pdf, support-sla.pdf

Answer generation failed; no answer score assigned.

### sla — review needed

Compare the response targets for Priority 1 and Priority 2 support.

Recall: 1.0; reciprocal rank: 1.0
Sources: support-sla.pdf, support-sla.pdf, disaster-recovery.pdf, security-policy.pdf

Answer generation failed; no answer score assigned.

### new-hire — review needed

When must a new hire finish security training?

Recall: 1.0; reciprocal rank: 1.0
Sources: onboarding.pdf, onboarding.pdf, security-policy.pdf, travel-policy.pdf

Answer generation failed; no answer score assigned.

### termination — review needed

How much notice is required to terminate the vendor contract?

Recall: 1.0; reciprocal rank: 1.0
Sources: vendor-contract.pdf, vendor-contract.pdf, procurement.pdf, procurement.pdf

Answer generation failed; no answer score assigned.

### backup-vs-recovery — review needed

What is backup retention and what are the recovery time and recovery point objectives?

Recall: 1.0; reciprocal rank: 1.0
Sources: disaster-recovery.pdf, security-policy.pdf, disaster-recovery.pdf, support-sla.pdf

Answer generation failed; no answer score assigned.

### unknown — review needed

What is the CEO personal phone number?

Recall: None; reciprocal rank: None
Sources: support-sla.pdf, leave-policy.pdf, procurement.pdf, security-policy.pdf

Answer generation failed; no answer score assigned.

