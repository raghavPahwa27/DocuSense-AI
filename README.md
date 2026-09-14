# DocuSense AI – Agentic Enterprise Knowledge Assistant

A minimal, agentic RAG system that lets you upload enterprise PDFs and ask questions through a clean **Streamlit UI**. Powered by **LangChain**, **Google Gemini**, and **FAISS**.

---

## What Does It Do?

DocuSense AI lets you:
1. **Upload any PDF** through the browser UI.
2. The document is automatically chunked, embedded, and indexed in FAISS.
3. A **ReAct agent** with two tools answers your questions:

| Tool | When the agent uses it |
|---|---|
| `document_retrieval` | Fetching facts from the uploaded document |
| `calculator` | Performing math on retrieved numbers |

The agent decides on its own which tool(s) to call and in what order.

---

## Traditional RAG vs. Agentic RAG

### Traditional RAG (fixed pipeline)
```
User Query
  → Always retrieve documents
  → LLM generates answer
```
The retrieval step is **hard-wired**. The LLM cannot skip it, repeat it, or do math.

### Agentic RAG (this project)
```
User Query
  → Agent reasons: "What do I need?"
  → Agent selects a tool (or multiple tools, in any order)
  → Tool returns an observation
  → Agent can reason again and call another tool
  → Final Answer
```
The agent is **in control**. It can:
- Call `document_retrieval` once or multiple times.
- Call `calculator` after retrieving a number.
- Answer directly without calling any tool if the answer is already known.

This is the core difference: **the LLM drives the loop, not the pipeline.**

---

## Architecture

```
                 ┌─────────────────┐
                 │   Streamlit UI  │  ← PDF upload + Q&A interface
                 └────────┬────────┘
                          ↓
                  ┌───────────────┐
                  │ Document Load │  ← PyPDFLoader
                  └───────┬───────┘
                          ↓
                    Text Chunking       ← RecursiveCharacterTextSplitter
                          ↓
                  Gemini Embeddings     ← models/embedding-001
                          ↓
                     FAISS Store        ← in-memory similarity index
                          ↓
                 Document Retrieval Tool
                          ↓
                      ReAct Agent       ← Thought → Action → Observation loop
                     ↙           ↘
           Document Tool      Calculator Tool
                          ↓
                    Agent Executor
                          ↓
                     Final Answer
```

---

## Technology Roles

| Technology | Role |
|---|---|
| **Streamlit** | Browser UI for PDF upload and question-answering |
| **LangChain** | Orchestrates loaders, splitters, tools, agent, and executor |
| **Gemini Embeddings** | Converts text chunks into dense vectors for semantic search |
| **FAISS** | In-memory vector index; finds the most relevant chunks for any query |
| **Document Retrieval Tool** | Wraps the FAISS retriever as a callable tool the agent can invoke |
| **Calculator Tool** | Executes Python math expressions; gives the agent deterministic arithmetic |
| **ReAct Agent** | LLM that alternates Thought → Action → Observation until Final Answer |
| **Agent Executor** | Runs the agent loop, dispatches tool calls, feeds observations back |

---

## Installation

```bash
# 1. Clone / enter the project folder
cd "DocuSense AI"

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Google API key
cp .env.example .env
# Open .env and paste your key from https://aistudio.google.com/app/apikey
```

---

## Running

### Streamlit UI (recommended)
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501` — upload a PDF from the sidebar and start asking.

### CLI fallback
```bash
python app.py
```
Loads PDFs from the `data/` folder and runs a terminal Q&A loop.

---

## Example Multi-Step Query

**Upload:** an annual financial report PDF

**Query:**
```
What was the total revenue mentioned in the report, and what is 15% of it?
```

**Agent flow (visible in the "Agent Reasoning Trace" expander):**
```
Action:      document_retrieval
Input:       "total revenue"
Observation: "...Total revenue for FY2024 was $4,500,000..."

Action:      calculator
Input:       0.15 * 4500000
Observation: 675000.0

Final Answer: The total revenue was $4,500,000 and 15% of it is $675,000.
```

---

## Project Structure

```
DocuSense AI/
├── streamlit_app.py  ← Streamlit UI + all agent logic
├── app.py            ← CLI version (terminal Q&A loop)
├── requirements.txt  ← Python dependencies
├── .env.example      ← API key template
├── README.md         ← This file
└── data/
    └── sample.pdf    ← Sample document (CLI mode)
```

---

## Key Constants (in both files)

| Constant | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between consecutive chunks |
| `k` (retriever) | 4 | Number of chunks returned per query |
