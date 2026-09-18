# DocuSense AI – Agentic RAG Knowledge Assistant

A **minimal, interview-ready** Agentic RAG application. Upload enterprise PDFs and ask questions through a Streamlit UI. Powered by LangChain, Google Gemini, and FAISS.

---

## What Makes This "Agentic RAG"?

Traditional RAG always retrieves from the vector store, then passes results to the LLM. DocuSense AI uses a **ReAct agent** that *decides at runtime* which tool to use:

| Scenario | Tool chosen by agent |
|---|---|
| "What does the contract say about liability?" | `document_retrieval` |
| "What is 18% of the revenue figure in the report?" | `document_retrieval` → `calculator` |
| "Who are the top competitors of this company?" | `document_retrieval` → `web_search` |
| "What is the current corporate tax rate?" | `web_search` |

The agent can also chain tools across multiple steps and uses conversational memory so follow-up questions like *"What about 2023?"* are understood in context.

---

## Architecture

```
                  PDF Upload
                      ↓
               Document Loader (PyPDF)
                      ↓
               Text Splitter (Recursive, 1000 chars)
                      ↓
               Gemini Embeddings (gemini-embedding-001)
                      ↓
               FAISS Vector Store (persisted to disk)
                      ↓
     ┌────────────────────────────────────────────┐
     │           User Question                    │
     │               ↓                            │
     │  ConversationBufferWindowMemory (k=5)      │
     │               ↓                            │
     │           ReAct Agent                      │
     │    (hwchase17/react-chat prompt)            │
     │        /        |          \               │
     │       ↓         ↓           ↓              │
     │  document_  calculator  web_search         │
     │  retrieval              (DDGS)             │
     │       \         |          /               │
     │        \        ↓         /                │
     │         \  gemini-3.6-flash  /             │
     │          \       ↓          /              │
     │           Final Answer + Web Sources       │
     └────────────────────────────────────────────┘
```

### Document Processing (Synchronous)
When a PDF is uploaded, the Streamlit app processes it immediately with a spinner:
1. Load PDF pages (`PyPDFLoader`)
2. Split into 1000-character overlapping chunks
3. Attach metadata: `source` (filename), `page`, `chunk_index`
4. Generate Gemini embeddings
5. If FAISS index exists → `add_documents()`; otherwise → `from_documents()`
6. Save updated index to `faiss_store/` on disk

Multiple PDFs share one FAISS index, so the agent retrieves across all of them.

### Duplicate Detection
MD5 hash of file content is stored in session state. Re-uploading the same file within a session is a no-op.

### Conversational Memory
`ConversationBufferWindowMemory(k=5)` keeps the last 5 exchanges in context. The agent uses `{chat_history}` from the `react-chat` prompt so follow-up questions are resolved correctly.

### Web Search Evidence
`WebSearchTool.run()` calls DDGS directly and side-stores `{title, url, snippet}` for each result. After `agent.invoke()`, the Streamlit app reads this evidence and displays a **"🌐 Web Sources"** expander below the answer. The LLM never generates URLs — they come from the DDGS API response.

---

## Technology Roles

| Technology | Role |
|---|---|
| **Streamlit** | Browser UI — upload, processing spinner, Q&A, chat history |
| **LangChain** | Orchestrates loaders, splitters, agent, tools, memory |
| **Gemini (gemini-3.6-flash)** | LLM powering the ReAct agent |
| **Gemini Embeddings** | Converts text chunks to dense vectors |
| **FAISS** | Local vector store — persisted to `faiss_store/` |
| **ReAct Agent** | Drives Thought → Action → Observation loop |
| **document_retrieval** | Semantic search over all uploaded PDFs |
| **calculator** | Python REPL for math expressions |
| **web_search (DDGS)** | DuckDuckGo web search with source citation |
| **ConversationBufferWindowMemory** | Last 5 exchanges as chat context |

---

## Installation

```bash
# 1. Clone the repository
cd "DocuSense AI"

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Google API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

---

## Running

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser. That's the only command needed.

---

## Project Structure

```
DocuSense AI/
├── streamlit_app.py   ← UI + agent + synchronous ingestion (everything in one file)
├── app.py             ← CLI version for quick terminal testing
├── requirements.txt   ← Python dependencies (no Redis, no queues, no extras)
├── .env               ← GOOGLE_API_KEY (gitignored)
├── README.md
├── data/
│   └── sample.pdf     ← Sample document for CLI mode
├── uploads/           ← Created at runtime: saved PDFs
└── faiss_store/       ← Created at runtime: persisted FAISS index
```

---

## Example Queries

### Multi-step: Document + Calculator
```
What was the total revenue in the report, and what is 15% of it?
```
```
Thought: I need the revenue figure first.
Action: document_retrieval → "Total revenue FY2024: $4,500,000"
Action: calculator → 0.15 * 4500000 = 675000.0
Final Answer: Revenue was $4.5M. 15% of that is $675,000.
```

### Conversational follow-up
```
User:  What was the revenue in 2024?
Agent: $4,500,000.
User:  What about 2023?
Agent: [uses chat_history context] The 2023 revenue was $3,800,000.
```

### Web Search with cited sources
```
User:  What is the current corporate tax rate in India?
```
```
Action: web_search → "corporate tax rate India 2024"
Final Answer: The base corporate tax rate for domestic companies is 22%.
```
🌐 **Web Sources (3)** — displayed with title, domain, snippet, and clickable link.

---

## Key Constants

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between adjacent chunks |
| `k` (retriever) | 4 | Chunks returned per query |
| `k` (memory) | 5 | Conversation turns kept in LLM context |
