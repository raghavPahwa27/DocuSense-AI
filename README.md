# DocuSense AI – Agentic Enterprise Knowledge Assistant

A minimal, agentic RAG system that lets you ask questions over enterprise documents (PDFs, reports, policies) from the command line. Powered by **LangChain**, **Google Gemini**, and **FAISS**.

---

## What Does It Do?

DocuSense AI loads your PDF documents, indexes them in a local vector store, and spins up a **ReAct agent** with two tools:

| Tool | When the agent uses it |
|---|---|
| `document_retrieval` | Fetching facts from documents |
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
Documents (data/*.pdf)
        ↓
  Document Loading          ← PyPDFDirectoryLoader
        ↓
   Text Chunking            ← RecursiveCharacterTextSplitter
        ↓
  Gemini Embeddings         ← models/embedding-001
        ↓
  FAISS Vector Store        ← in-memory similarity index
        ↓
Document Retrieval Tool ──┐
                           ├──▶ ReAct Agent ──▶ Agent Executor ──▶ Final Answer
    Calculator Tool ───────┘
```

---

## Technology Roles

| Technology | Role |
|---|---|
| **LangChain** | Orchestrates the entire pipeline: loaders, splitters, tools, agent, executor |
| **Gemini Embeddings** | Converts text chunks into dense vectors for semantic similarity search |
| **FAISS** | In-memory vector index; finds the most relevant chunks for any query |
| **Document Retrieval Tool** | Wraps FAISS retriever as a callable tool the agent can invoke |
| **Calculator Tool** | Executes Python math expressions; gives the agent deterministic arithmetic |
| **ReAct Agent** | LLM that alternates Thought → Action → Observation until it has a Final Answer |
| **Agent Executor** | Runs the agent loop, dispatches tool calls, feeds observations back to the agent |

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

# 5. Add PDFs
# Drop one or more PDF files into the data/ folder.
```

---

## Running

```bash
python app.py
```

The system will load and index your PDFs, then show a prompt:

```
============================================================
  DocuSense AI – Agentic Enterprise Knowledge Assistant
  Type your question and press Enter. Type 'exit' to quit.
============================================================

You: 
```

---

## Example Multi-Step Query

**Query:**
```
What was the total revenue mentioned in the annual report, and what is 15% of it?
```

**Agent flow:**
```
Thought: I need to find the revenue figure. I'll use document_retrieval.
Action: document_retrieval
Action Input: "total revenue annual report"
Observation: "...Total revenue for FY2024 was $4,500,000..."

Thought: Now I need to calculate 15% of 4,500,000.
Action: calculator
Action Input: 0.15 * 4500000
Observation: 675000.0

Thought: I have both pieces of information.
Final Answer: The total revenue was $4,500,000 and 15% of it is $675,000.
```

---

## Project Structure

```
DocuSense AI/
├── app.py            ← All application logic (load, embed, agent, CLI)
├── requirements.txt  ← Python dependencies
├── .env.example      ← API key template
├── README.md         ← This file
└── data/
    └── sample.pdf    ← Add your PDFs here
```

---

## Key Constants (in app.py)

| Constant | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between consecutive chunks |
| `k` (retriever) | 4 | Number of chunks returned per query |
