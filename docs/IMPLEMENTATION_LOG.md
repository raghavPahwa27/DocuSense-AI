# Implementation and verification log

## Preflight — 2026-10-12

- Original UI: Streamlit, PyPDFLoader, recursive splitting (1000/150), normalized Gemini embeddings and FAISS IndexFlatIP, LangChain ReAct/AgentExecutor, PythonREPLTool calculator, DDGS evidence, five-turn ConversationBufferWindowMemory.
- Existing FAISS index: 46 vectors, 3072 dimensions. Read using FAISS only; no pickle deserialization.
- Fresh installed `langchain-google-genai==2.0.10` document and query embedding calls both returned 3072 values for `models/gemini-embedding-001`. Neither supplies output dimensionality. Requirements previously specified 2.1.12, unlike the installed package.
- Existing Pinecone index validated: dense, cosine, dimension 3072, ready, serverless AWS us-east-1, no integrated embeddings. No index created or reconfigured.
- Installed Pinecone 7.3.0, the latest available to the existing Python 3.9 resolver; current Pinecone docs require Python >=3.10 for the newest SDK. Python 3.9/LibreSSL produce upstream support warnings.
- FAISS, uploads, and original PDF hashes captured before migration; no existing file removed.

## Stage 1

Files: shared `docusense/config.py`, `store.py`, `agent.py`; Streamlit and CLI integration; `migrate_to_pinecone.py`; initial mock fixtures and storage tests.

- Deterministic SHA-256 document IDs, ordinal chunk IDs, revision-specific vector IDs, original chunk sizes, text and citation metadata.
- Namespace and metadata owner scoping. A private SQLite publication ledger excludes partial ingestion, stale revisions and deleted documents from retrieval.
- Re-embed source PDFs; verify metadata via fetch and retrieval visibility before publication. Re-index publishes after successful verification; old revisions remain hidden until document deletion.
- Nine mocked tests passed. Bytecode compilation initially hit the macOS cache sandbox path; rerun with `PYTHONPYCACHEPREFIX=/tmp/docusense-pycache` succeeded. `pip check` passed.
- Live migration outcome recorded separately after completion. Uploaded legacy hashed filenames cannot be mapped back to original names without a user-supplied mapping; use `--source` to restore a known filename safely.

Live ingestion initially exposed an SDK compatibility bug: Pinecone 7.3 returns a dataclass for fetch, unlike its describe/query objects. Added dataclass response handling and a regression test (10 tests pass); retries reuse deterministic vector IDs.

Final Stage 1 verification: all three available original PDFs successfully migrated (1 + 3 + 34 = 38 chunks) and retrieved. Original FAISS holds 46 vectors; the available PDFs do not establish complete coverage of every historical FAISS record. All six original source/index files retained identical SHA-256 hashes. Final SDK regression suite: 11 passed.

## Stage 2

Files: `store.py`, `agent.py`, new `calculator.py`, Streamlit management controls, storage/agent tests.

- Private random Streamlit session ID; no global owner override or cross-session cached executor. CLI/migration owners are explicit and for trusted local operators.
- Namespace, query/delete filters, active document revisions and response metadata checks enforce application isolation. Duplicate PDFs remain duplicates after filename changes.
- Safe messages for PDF parsing, empty text, missing files, invalid vectors, embedding/network failures, missing configuration and empty results. 25 MB file limit. Original PDFs retained on deletion.
- Scoped delete/re-index buttons clear conversation memory/history. Failed remote deletion remains hidden and retryable.
- HTML-escape dynamic text in existing cards. Preserve calculator through PythonREPLTool with a bounded arithmetic AST gate: arbitrary Python would otherwise bypass isolation and access process secrets.
- Security limits: session IDs are not authenticated identities; no authorization system. New browser sessions do not inherit old documents. Anyone with process/filesystem/API-key access can bypass application filters. Pinecone stores chunk text; Gemini receives documents and queries; reranking sends candidates to Pinecone inference. Public web queries leave the app. Prompt injection remains a risk. Local state is plaintext, for one durable local deployment; coordinate CLI writers externally and do not deploy multiple independent state directories against the same owner. No claim of production readiness.

## Stage 3

Files: `evaluation/{dataset.json,metrics.py,run.py,README.md}`, synthetic baseline JSON/Markdown reports, evaluation metric tests.

- Baseline established before reranker implementation: real Gemini embeddings + Pinecone vector retrieval, Recall@4 **1.0000**, MRR@4 **0.9444**, nine answerable queries plus one abstention case, 24 synthetic chunks. Zero retrieval errors.
- Initial answer/judge run hit Gemini's 5 requests/minute free-tier quota: 3 answers/judgments succeeded, 7 answer generations failed. Successful automated relevance/support judgments averaged 1.0, but are partial, biased automated evidence, not manual verification. All failures are retained in the baseline report. Added 13-second configurable pacing for subsequent runs.
- 26 tests passed. Report includes exact retrieved IDs, contexts, answer points/proxy failures, model/dimension/dataset hash, API failure counts and null manual judgments. No private PDF content is in the test dataset or reports.

## Stage 4

Files: `docusense/retrieval.py`, configuration and agent/UI integration, evaluation comparison options, reranker tests and reports.

- Pinecone SDK 7.3.0 signatures inspected. Live `list_models(type='rerank')` confirmed `bge-reranker-v2-m3`, `pinecone-rerank-v0`, `cohere-rerank-3.5` (deprecated), and `cohere-rerank-4-fast`. Selected configurable `bge-reranker-v2-m3` for initial evaluation; no integrated embedding calls.
- Query 20 Gemini-vector candidates, hosted Inference rerank to 4; preserve original context and citation/owner metadata by mapping validated response indices. Configurable vector-only disable and marked fallback for unavailable models, API failures or malformed results.
- Actual 20-to-4 run: Recall@4 **1.0000**, MRR@4 **1.0000**, zero API errors/fallbacks, same dataset as baseline. MRR delta +0.0556, no Recall improvement. Reranking stays disabled by default: this tiny synthetic sample does not justify a general quality claim.
- 34 tests passed. Candidate count 16 evaluated separately; answer evaluation being repeated with pacing because of the observed Gemini quota.

The 16-to-4 live trial also returned Recall@4 1.0000 / MRR@4 1.0000 with zero fallbacks. Kept initial configurable 20-to-4 default; no evidence this tiny corpus identifies optimal settings.

## Stage 5

Files: requirements and optional dev/legacy requirement files, `.env.example`, `.gitignore`, README, expanded configuration/agent/UI/evaluation tests, report documentation.

- Every direct requirement exactly matches the installed version; `pip check` passes. Python compilation and whitespace checks pass. FAISS removed from active requirements/imports but left installed and available through an optional legacy requirements file. No unrelated framework, queue or worker added.
- Real LangChain ReAct/AgentExecutor tested with a deterministic mock LLM: document tool → calculator → cited answer; follow-up memory retained; five-turn window and clear behavior verified.
- Streamlit AppTest covered separate owners, stable owner across reruns, empty state, submit, document-source display and clearing chat. Initial test failure involved a selectbox formatter reading session state outside script context; changed it to capture a stable name mapping, then passed.
- Actual LangChain Hub prompt loaded; required variables and Final Answer parser format verified. Actual DuckDuckGo returned zero results; graceful empty response verified. Nonempty web-source capture/deduplication verified with mocks, not claimed as a live positive web result.
- Paced Gemini run: seven answers, six automated judgments, three answer errors and one judge error when the **20 requests/day** free-tier limit was reached. Report: `evaluation/reports/vector-paced.json`. Successful judgment means are 1.0, but partial and not independently human verified. Live Gemini ReAct chat and a full reranked-answer comparison cannot be completed with this quota today. No LLM model switch made. Added a quota circuit breaker for future evaluation runs.
- Manual actions: after the Gemini quota resets or the account quota changes, repeat the documented answer/judge and interactive checks. Review private-document answer/citation correctness, supply missing original PDFs/names if the historical 46 FAISS vectors must be fully accounted for, and choose whether to enable reranking on domain evidence. No production-readiness claim.

Final checks: **40 tests passed**, including real AgentExecutor with mocked LLM and Streamlit AppTest. Actual Pinecone synthetic smoke checks all passed: persisted duplicate no-op, two-owner isolation, verified re-index, deletion hidden from retrieval, and the other owner's document surviving deletion. Only disposable synthetic fixtures were cleaned up. `pip check`, compilation, and `git diff --check` passed. All six original PDF/FAISS files remained byte-for-byte unchanged.

## Follow-up: current-date grounding and web-search abstention

The observed weekday error came from a failed web search followed by an unconstrained model answer. Added built-in `datetime`/`zoneinfo` context for Asia/Kolkata to every agent invocation, plus a deterministic direct response for simple current date/weekday questions so they never invoke web search. The existing prompt now instructs the model to use the trusted timestamp, avoid unnecessary tools, distinguish document evidence, and abstain when evidence is insufficient. Empty and failed web searches now return explicit non-evidence markers instead of ambiguous prose. No tools, APIs, or dependencies were added. **50 tests pass.**

Saved reports were subsequently rendered with successful-score counts derived from their existing case results and expected points copied from the versioned dataset; measurements were not invented or replaced. Full live Gemini agent/answer verification and human citation review remain explicitly incomplete because of account quota and required human review.

## Follow-up: Groq chat provider

User selected Groq `llama-3.3-70b-versatile`. Added `langchain-groq==0.2.5`, a shared chat-model factory, provider/key/model settings, dynamic UI model display, and matching evaluation-provider metadata. Stored the supplied Groq credential only in the ignored local `.env`; retained the separate Google embedding credential. No embedding model or Pinecone vectors changed. Updated setup documentation and provider-routing/configuration tests.

Verification: 43 tests passed, compilation and whitespace checks passed; dependency check passed after installing the compatible adapter. The running browser displays `llama-3.3-70b-versatile` for chat and `gemini-embedding-001` for embeddings. Live Groq authentication/inference is not verified: the requested network test was not executed because automatic approval review hit an account usage limit, not because the action was deemed unsafe.

## Follow-up: unavailable Groq model and GPT-OSS tool compatibility

A live request returned 404 `model_not_found` for `llama-3.3-70b-versatile`; the credential successfully listed available models. The user selected `openai/gpt-oss-20b`. Normal requests worked, but the original text-only ReAct setup triggered Groq's `Tool choice is none, but model called a tool` error. Registered the existing local tools for GPT-OSS and added a validated single-call-to-ReAct adapter. `create_react_agent`, AgentExecutor, local calculator restrictions, document filtering and memory are preserved.

Live verification: GPT-OSS used the calculator for 17 * 23, returned 391, and saved the exchange in memory. Added provider-specific redacted errors, configuration-aware cached-agent rebuilding, and adapter regression tests. **47 tests passed.** Restarted Streamlit to reload `.env`; session reset does not delete saved PDFs or Pinecone data. Gemini embeddings remain unchanged.

## Follow-up: interview-friendly architecture refactor

Created `codex/simplify-docusense` from the working Pinecone version. The active application now has six main modules: the Streamlit UI plus `agent.py`, `rag.py`, `tools.py`, `document_manager.py`, and `config.py`. The small `llm.py`, `errors.py`, `calculator.py`, `react_adapter.py`, `retrieval.py`, and `store.py` modules were absorbed into their related modules. The date-only executor wrapper was replaced by `ask_agent`; the source-capture helper is now in `tools.py`. `app.py` remains a small CLI using the same code paths. Historical migration scripts were moved to `scripts/archive/`.

The SQLite manifest, deterministic revision IDs, verification before publication, hidden pending deletion, session namespaces, owner filters, and returned-metadata checks remain because they prevent partial uploads, stale revisions, failed deletions, and cross-session results from becoming visible. No live Pinecone data was changed during this refactor. Baseline and final suites each had **55 passed, 0 failed, 0 skipped**. `pip check` passed; the CLI and archived migration script help commands worked; Streamlit started on a temporary local port and its health endpoint returned HTTP 200. Live model answer quality and private citation correctness were not re-evaluated by these checks.

## Follow-up: multi-part question coverage

The agent prompt now requires identifying and answering every distinct part of a user message, using numbered points or clear sections in the original order. Document-specific parts continue through `document_retrieval`, while general questions can be answered directly; unsupported parts must be called out while the answer continues for supported parts. The date shortcut is restricted to date-only questions so a mixed document/date question is delegated to the normal ReAct flow with the trusted time context. **55 tests passed.**
