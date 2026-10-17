# DocuSense AI — Agentic Knowledge Assistant

DocuSense AI is an agentic RAG application that lets users upload PDF documents and ask questions in natural language. It combines semantic search, large language models, and an autonomous tool-using agent to answer questions with source citations.

## Features

- **PDF Question Answering:** Upload PDFs and ask questions about their content.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document chunks using Gemini embeddings and Pinecone vector search.
- **ReAct Agent:** Uses LangChain's ReAct agent and AgentExecutor to select tools based on the user's question.
- **Multiple Tools:** Document retrieval, arithmetic calculator, and DuckDuckGo web search.
- **Source Citations:** Displays document sources, page numbers, and chunk information.
- **Conversation Memory:** Maintains a five-turn conversation window.
- **Document Management:** Supports duplicate detection, re-indexing, and deletion.
- **Session Isolation:** Separates document retrieval by session using Pinecone namespaces and metadata filters.
- **Optional Reranking:** Supports Pinecone-hosted reranking to reorder retrieved chunks before generation.
- **Flexible Chat Models:** Supports Gemini and Groq for answer generation.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Agent Framework | LangChain |
| Agent Architecture | ReAct + AgentExecutor |
| Chat Models | Google Gemini / Groq |
| Embeddings | Gemini `gemini-embedding-001` |
| Vector Database | Pinecone |
| Document Processing | PyPDF + Recursive Character Text Splitter |
| Metadata and Revision Tracking | SQLite |
| Web Search | DuckDuckGo Search |

## How It Works

### 1. Document Ingestion

1. The user uploads a PDF through Streamlit.
2. The application validates the document and checks for duplicates.
3. The PDF is extracted and split into chunks of 1,000 characters with 150 characters of overlap.
4. Gemini converts the chunks into embeddings.
5. The embeddings and associated metadata are stored in Pinecone.
6. The application verifies the stored vectors before marking the document revision as active in its SQLite manifest.

### 2. Question Answering

1. The user submits a question.
2. The LangChain ReAct agent determines which tool is appropriate.
3. For document questions, the retrieval tool embeds the query and searches Pinecone for relevant chunks.
4. Retrieved evidence is passed to the language model to generate an answer.
5. The application displays the response along with available source citations and tool traces.

The agent can also use the calculator for arithmetic and attempt web searches for questions requiring external information.

### Architecture Overview

```text
PDF Upload
    |
    v
Validation and Duplicate Check
    |
    v
PDF Extraction and Chunking
    |
    v
Gemini Embeddings
    |
    v
Pinecone Vector Database
    |
    v
Verified Revision + SQLite Manifest


User Question
    |
    v
LangChain ReAct Agent
    |
    +---- Document Retrieval --> Pinecone
    |
    +---- Calculator
    |
    +---- DuckDuckGo Web Search
    |
    v
LLM Answer Generation
    |
    v
Answer + Available Citations
```

## Getting Started

### Prerequisites

- Python 3.11 recommended
- A Google AI API key
- A Pinecone API key and an existing compatible vector index
- A Groq API key if using Groq for chat

### 1. Clone the Repository

```bash
git clone https://github.com/raghavPahwa27/DocuSense-AI.git
cd DocuSense-AI
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_existing_index_name
```

The Pinecone index must be a compatible dense-vector index using cosine similarity and the correct embedding dimension (3072 for the configured Gemini embedding model).

**Optional: Use Groq for chat**

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Keep `GOOGLE_API_KEY` configured because Gemini is still used for embeddings.

To use Gemini for chat, set:

```env
LLM_PROVIDER=gemini
```

**Optional: Enable reranking**

```env
RERANK_ENABLED=true
PINECONE_RERANK_MODEL=bge-reranker-v2-m3
RETRIEVAL_CANDIDATES=20
RETRIEVAL_TOP_K=4
```

Reranking is disabled by default.

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed by Streamlit in your browser.

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Check dependency consistency:

```bash
pip check
```

The latest reported test run passed **56 tests with 0 failures and 0 skipped tests**. Mocked tests verify application behavior, but they do not guarantee that external services or live model responses work correctly.

## Project Structure

```text
DocuSense-AI/
├── docusense/
│   ├── agent.py             # ReAct agent, LLM configuration and memory
│   ├── rag.py               # Embeddings, Pinecone retrieval and reranking
│   ├── tools.py             # Retrieval, calculator and web-search tools
│   ├── document_manager.py  # Ingestion, revisions and document deletion
│   └── config.py            # Configuration and credentials
├── evaluation/              # RAG evaluation and reports
├── tests/                   # Automated tests
├── streamlit_app.py         # Streamlit user interface
├── app.py                   # CLI entry point
├── migrate_to_pinecone.py   # Legacy document migration utility
├── requirements.txt
├── requirements-dev.txt
└── README.md
```


---

**Repository:** [RaghavPahwa27/DocuSense-AI](https://github.com/raghavPahwa27/DocuSense-AI)
