# DocuSense AI – Agentic Enterprise Knowledge Assistant

A minimal, agentic RAG system for uploading enterprise PDFs and asking questions through a **Streamlit UI**. Powered by **LangChain**, **Google Gemini**, **FAISS**, **Redis**, and **RQ**.

---

## What Does It Do?

DocuSense AI lets you:
1. **Upload multiple PDFs** — each is processed in the background (no UI blocking).
2. Documents are chunked, embedded, and added to a **shared FAISS index** on disk.
3. A **ReAct agent** with three tools answers your questions with **conversational memory**.

| Tool | When the agent uses it |
|---|---|
| `document_retrieval` | Fetching facts from any uploaded document |
| `calculator` | Performing math on retrieved numbers |
| `web_search` | Searching the public web via DuckDuckGo |

The agent decides which tool(s) to call and in what order.
Chat memory (last 5 exchanges) allows follow-up questions like *"What about 2023?"*

---

## Architecture

```
Streamlit UI
    /           \
Document Upload  Chat (with Memory)
    ↓                   ↓
Redis + RQ        ConversationBufferWindowMemory(k=5)
    ↓                   ↓
RQ Worker         ReAct Agent  ← hwchase17/react-chat prompt
    ↓              /    |    \
Load PDF      FAISS  Calc  DuckDuckGo
    ↓              \    |    /
Chunk + Embed        LLM
    ↓                 ↓
faiss_store/       Answer
(disk, shared)
```

### Multi-document FAISS
- All documents share **one** FAISS index saved to `faiss_store/` on disk.
- A `FileLock` prevents concurrent writes from multiple RQ workers corrupting the index.
- Each chunk carries metadata: `document_name`, `page_number`, `chunk_id`.
- Duplicate detection: MD5 file hash stored in Redis — same file is never re-processed.

### Chat Memory
- `ConversationBufferWindowMemory(k=5)` — only the last 5 exchanges are sent to the LLM.
- Memory persists when the agent is rebuilt (new doc completes) — conversations continue seamlessly.
- "Clear Chat" button clears both the display and the memory.

### Background Processing (Redis + RQ)
Document ingestion status:
```
Upload → queued 🕐 → processing ⏳ → completed ✅
                                  → failed    ❌
```
Status is tracked per document hash in Redis. The Streamlit sidebar shows live status with a Refresh button.

---

## Technology Roles

| Technology | Role |
|---|---|
| **Streamlit** | Browser UI for PDF upload, status display, and Q&A |
| **Redis** | Job queue backend + document status store |
| **RQ (Redis Queue)** | Runs `ingest_document` jobs in a background worker |
| **FileLock** | Prevents concurrent worker writes from corrupting the FAISS index |
| **LangChain** | Orchestrates loaders, splitters, tools, agent, and executor |
| **Gemini Embeddings** | Converts text chunks into dense vectors |
| **FAISS** | Persisted on-disk vector index for all documents |
| **Document Retrieval Tool** | Wraps FAISS retriever; returns chunks with doc name + page |
| **Calculator Tool** | Executes Python math expressions |
| **Web Search Tool** | Queries DuckDuckGo for public/current information |
| **ConversationBufferWindowMemory** | Keeps last 5 exchanges for follow-up questions |
| **ReAct Agent** | Drives Thought → Action → Observation loop |
| **Agent Executor** | Runs the agent loop with memory + tool dispatch |

---

## Installation

```bash
# 1. Clone and enter the project folder
cd "DocuSense AI"

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Google API key
# Edit .env and add: GOOGLE_API_KEY=your_key_here
```

---

## Running

### Terminal 1 — Redis
```bash
redis-server
```

### Terminal 2 — RQ Worker
```bash
cd "DocuSense AI"
source venv/bin/activate
rq worker
```

### Terminal 3 — Streamlit
```bash
cd "DocuSense AI"
source venv/bin/activate
streamlit run streamlit_app.py
```

### CLI Mode (no Redis required)
```bash
source venv/bin/activate
python app.py
```
Reads PDFs from `data/`, runs a terminal Q&A loop.

---

## Example Queries

### Multi-step: Document + Calculator
```
What was the total revenue mentioned in the report, and what is 15% of it?
```
```
Action: document_retrieval → "Total revenue for FY2024 was $4,500,000"
Action: calculator         → 0.15 * 4500000 = 675000.0
Final Answer: Revenue was $4.5M; 15% is $675,000.
```

### Follow-up with Memory
```
User:  What was the revenue in 2024?
Agent: The revenue in 2024 was $4,500,000.
User:  What about 2023?
Agent: [uses chat_history to understand "2023" is about revenue] ...
```

### Web Search: External knowledge
```
What is the current corporate tax rate in India?
```
```
Action: web_search → "corporate tax rate India 2024"
Final Answer: The base corporate tax rate is 22% for domestic companies.
```

### Multi-doc + Web Search
```
What industry does this company operate in, and who are its top competitors?
```
```
Action: document_retrieval → "Acme Corp is an enterprise software company"
Action: web_search         → "top enterprise software competitors 2024"
Final Answer: Acme operates in enterprise software; top competitors include SAP, Oracle, Microsoft.
```

---

## Project Structure

```
DocuSense AI/
├── streamlit_app.py  ← Streamlit UI + agent logic (multi-doc, memory, status)
├── worker.py         ← RQ job: PDF → chunk → embed → FAISS (background)
├── app.py            ← CLI version (no Redis, reads from data/)
├── requirements.txt  ← Python dependencies
├── .env              ← API key (gitignored)
├── README.md         ← This file
├── data/
│   └── sample.pdf    ← Sample document for CLI mode
├── uploads/          ← Created at runtime — PDFs saved here for worker
└── faiss_store/      ← Created at runtime — persisted FAISS index
```

---

## Key Constants

| Constant | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `k` (retriever) | 4 | Chunks returned per query |
| `k` (memory) | 5 | Conversation turns kept in context |
