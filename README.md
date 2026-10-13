# DocuSense AI

Upload PDFs and ask questions through the existing Streamlit interface. Gemini powers embeddings; Groq or Gemini powers answers; a LangChain ReAct agent uses document retrieval, an arithmetic calculator, DuckDuckGo search, five-turn conversation memory, and source citations. Pinecone now stores vectors. No workers or additional web framework are required.

## Setup

The tested repository environment is Python 3.9.6 with the pinned packages below. Python 3.9 and its macOS LibreSSL runtime emit upstream support warnings; use Python 3.11 for a new environment and run the checks before deployment. The migration deliberately retains LangChain 0.3 and the installed Gemini integration rather than moving to LangChain 1.x. `langchain-google-genai` is pinned to the **actually installed/tested 2.0.10**; the old requirements listed an uninstalled 2.1.12. Pinecone 7.3.0 is compatible with this existing environment; newer SDKs require newer Python.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env  # only for a fresh setup; do not overwrite existing credentials
streamlit run streamlit_app.py
```

Set `GOOGLE_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME` locally. Environment variables take precedence over Streamlit secrets; `.env` is loaded without overwriting existing environment variables. Streamlit also accepts the same top-level names in `.streamlit/secrets.toml`. Do not commit either secret file. Restart Streamlit after changing retrieval settings.

The Gemini key needs access/quota for **`models/gemini-embedding-001`**. For Groq chat, set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and `GROQ_MODEL=openai/gpt-oss-20b`. Keep `GOOGLE_API_KEY` for embeddings; do not replace it with a Groq key. Set `LLM_PROVIDER=gemini` to use the original `gemini-3.6-flash` chat model. The app does not silently change models if access fails. Provider failures produce redacted messages; check your provider console for account-specific details. Keep provider HTTP debug logging and LangSmith tracing off for confidential documents unless explicitly approved.

## Pinecone index configuration

Use an existing **standard vector index** with:

| Setting | Required value |
|---|---|
| Vector type | dense |
| Metric | cosine |
| Dimension | Actual Gemini output; measured **3072** for document and query calls in this environment |
| Hosted/integrated embedding model | None |
| Filterable metadata | `owner_id`, `document_id`, `revision` (default all-field indexing is suitable) |
| Storage | Serverless recommended; the validated existing index uses AWS `us-east-1` |

Startup probes the existing embedding configuration and validates dimension, metric, readiness, external-vector configuration, and any selective metadata schema returned by the SDK before writing. Both documents and queries must match the measured dimension. Do not create a Pinecone-hosted embedding index. Cloud and region are operator decisions; the app never creates, replaces, deletes, or reconfigures an index automatically.

Each session uses its own hashed namespace. Metadata includes text, SHA-256 document ID, stable ordinal chunk ID, revision, source filename, zero-based page, chunk index, and owner ID. Citations display **one-based page numbers**. Original chunking remains `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`.

## Safe migration from FAISS

`faiss_store/` and original PDFs remain untouched. The new application never deserializes FAISS pickle files. Original PDFs must be parsed and re-embedded; a FAISS file is not copied into Pinecone.

```bash
# Select only PDFs that belong to this trusted local owner.
python migrate_to_pinecone.py --owner-id YOUR_PRIVATE_LOCAL_OWNER --pdf data/sample.pdf
# Restore a known original filename when a legacy upload has a hashed filename:
python migrate_to_pinecone.py --owner-id YOUR_PRIVATE_LOCAL_OWNER --pdf uploads/HASH.pdf --source original-report.pdf
# Explicit re-indexing after verification of new vectors:
python migrate_to_pinecone.py --owner-id YOUR_PRIVATE_LOCAL_OWNER --pdf data/sample.pdf --reindex
# CLI with the same owner and the existing tools/memory:
python app.py --owner-id YOUR_PRIVATE_LOCAL_OWNER
```

For the migration performed in this workspace, the private owner is stored in `.docusense/migration-owner.txt`, and the verification report in `.docusense/migration-report.json`. Use that owner with the trusted CLI to access the migrated originals. It is never automatically attached to a browser session. Re-upload a PDF to use it in a new UI session.

Three available originals produced **38 verified chunks**. The historical FAISS index contains **46 vectors**: the available originals cannot establish complete coverage of all historical content. Hashed legacy upload filenames lack an original-name mapping. Supply missing originals/citation names if needed. Preserve FAISS until you have manually checked representative private questions and citations. The old `migrate_faiss_to_cosine.py` is a legacy utility that overwrites a FAISS index; **do not run it for Pinecone migration**. It remains unchanged for historical reference.

Ingestion embeds in bounded batches, upserts deterministic IDs, verifies fetched metadata plus query visibility, then activates the revision in a private SQLite manifest. A failure before publication leaves staged vectors excluded from retrieval; retry safely. Duplicate detection uses document bytes and persists across restarts for the same local owner. Re-indexing activates the replacement only after verification; previous revisions remain stored but excluded, and scoped document deletion removes all revisions. There are no multi-service transactions or distributed writer locks: do not concurrently mutate the same owner/document from separate processes.

Keep `.docusense/` durable and private. Losing it prevents retrieval of previously published vectors (fail closed); restore the manifest or re-ingest approved originals. Pinecone alone is not the document catalog in this local design. No automatic cleanup of unverified vectors or abandoned sessions runs in the background.

## Document isolation and controls

Streamlit creates an unpredictable session ID on first use and preserves it across reruns. It does not read a global owner from `.env`, accept an owner from a URL, or reuse another session's agent/store. New browser sessions do not recover old session identities. CLI owners are explicit identifiers for a trusted local operator, **not credentials or proof of authorization**.

All retrieval and deletion use an owner namespace and metadata filters; retrieval additionally filters active document revisions and checks returned metadata. Empty results are reported explicitly. “Manage documents” supports re-indexing and deleting only documents registered to that session. Those actions clear conversation memory/history to avoid stale context. Delete retains the original local PDF; a remote failure leaves a hidden, retryable pending deletion. Pinecone deletion is eventually consistent; a successful request does not promise immediate physical erasure. A still-selected upload is remembered so a UI rerun does not undo deletion; start a new session to upload it again.

PDFs larger than 25 MB, invalid/unreadable PDFs, and documents without extractable text are rejected. Scanned PDFs need OCR outside this app. User-visible errors do not contain provider bodies, keys or document text. The UI continues to show tool traces and citations to its current session; those contain document information and should not be shared indiscriminately.

## Reranking

```dotenv
RERANK_ENABLED=false
PINECONE_RERANK_MODEL=bge-reranker-v2-m3
RETRIEVAL_CANDIDATES=20
RETRIEVAL_TOP_K=4
```

When enabled, a normal Pinecone `query` retrieves Gemini vectors, then **`pc.inference.rerank`** orders candidate text and returns the top four. The model is checked against `pc.inference.list_models(type='rerank')`. No integrated embedding workflow is used. IDs and citation metadata stay attached to the original chunks. Missing models, API failures or malformed rankings fall back to vector order; the UI and evaluation report identify fallback. Empty candidates bypass inference. Settings require `1 <= top_k <= candidates <= 100`.

Live synthetic retrieval results:

| Retrieval | Recall@4 | MRR@4 | Rerank fallbacks |
|---|---:|---:|---:|
| Vector only | 1.0000 | 0.9444 | — |
| 20 candidates → rerank 4 | 1.0000 | 1.0000 | 0 |
| 16 candidates → rerank 4 | 1.0000 | 1.0000 | 0 |

This is a small synthetic benchmark, not proof of better enterprise answers. Reranking remains disabled by default; assess quality, latency and cost on approved domain questions before enabling it. The 16-candidate trial tied the 20-candidate trial, which does not establish an optimum. Keep 20 as the initial configurable setting.

## Evaluation and tests

See [evaluation/README.md](evaluation/README.md) for definitions, reproducibility and report limitations.

```bash
python -m pytest -q
pip check
python -m evaluation.run --ingest --output evaluation/reports/vector-baseline.json
python -m evaluation.run --mode rerank --candidates 20 --compare-with evaluation/reports/vector-baseline.json --output evaluation/reports/rerank-20.json
# Slower: live answer generation and automated relevance/support judgments.
python -m evaluation.run --answers --judge --llm-interval 15 --output evaluation/reports/vector-paced.json
```

Evaluation runs independently of Streamlit in a synthetic-dataset namespace. Reports contain Recall@K, MRR@K, per-case examples/failures, answer-point lexical checks, optional configured-chat-model relevance/support judgments, API failures, successful score counts and explicit null manual judgments. Gemini's observed free-tier chat quota was 5 requests/minute; pacing reduces quota failures. Automated lexical and model-based judgments must not be presented as human-verified correctness. Private-corpus reports contain context text; write them under `evaluation/private-reports/` or another protected, ignored directory.

## Remaining limitations

- Session isolation is not authentication or authorization. Add verified user identities and access control before multi-user deployment. Server/filesystem/Pinecone-key access can bypass application filters. The local manifest is plaintext, with private file permissions, and is designed for a single durable deployment.
- Gemini receives embedding text/queries. The selected chat provider (Groq or Gemini) receives agent context and conversation history. Pinecone stores text metadata, hosted reranking receives candidates, and DuckDuckGo receives web queries. Assess provider data policies for your own documents.
- Retrieved PDFs and web content can contain prompt injection. The calculator retains PythonREPLTool behind an arithmetic-only AST gate to prevent arbitrary code/file access; there is no broader sandbox or prompt-injection defense guarantee.
- Retention, access auditing, authenticated deletion, OCR, malware scanning, durable user accounts and multi-process write coordination are not implemented. Hidden partial/old vectors consume storage until scoped cleanup or document deletion. Session expiration does not delete remote data automatically.
- ReAct can select the wrong tool or omit an answer citation; independently captured source/page/chunk evidence remains visible. LLM relevance judgments can be wrong. Review private-question answers manually.
- Migration success does not imply production readiness.

Project code is split into `docusense/` (shared services/tools), `streamlit_app.py` (existing UI), `app.py` (CLI), `migrate_to_pinecone.py`, `evaluation/`, and `tests/`. Historical FAISS support is optional in `requirements-legacy-faiss.txt` and is not used by the active application.

Official references: [Pinecone indexes](https://docs.pinecone.io/guides/index-data/create-an-index), [namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces), [hosted reranking](https://docs.pinecone.io/guides/search/rerank-results), [SDK compatibility](https://docs.pinecone.io/reference/sdks/python/overview). Detailed outcomes and limitations are in [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md).

Final local verification: **40 tests passed**, dependency checks passed, original-file hashes unchanged, and live Pinecone duplicate/isolation/re-index/delete fixture checks passed. Full live Gemini agent and answer comparison remain blocked by the observed daily quota; mocked ReAct/UI tests are distinguished from live model validation in the implementation log.

The earlier checked-in benchmark reports used Gemini chat. New evaluations use the configured chat provider and record its model/provider. Do not compare answer scores across providers as if only reranking changed. Changing the chat provider does not require re-indexing. Restart Streamlit after changing `.env` so existing agents use the new configuration.
