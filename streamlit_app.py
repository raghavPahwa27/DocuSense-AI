import os
import hashlib
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from duckduckgo_search import DDGS as _DDGS
from langchain.memory import ConversationBufferWindowMemory

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_DIR = str(Path(__file__).parent / "faiss_store")
UPLOADS_DIR     = str(Path(__file__).parent / "uploads")
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocuSense AI", page_icon="🧠", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

.hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #ff7b72 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.hero-sub { color: #8b949e; font-size: 0.95rem; margin-top: 0.25rem; font-weight: 400; }

.stat-row { display: flex; gap: 12px; margin-top: 1rem; flex-wrap: wrap; }
.stat-badge {
    background: #21262d; border: 1px solid #30363d;
    border-radius: 8px; padding: 8px 14px; font-size: 0.82rem; color: #8b949e;
}
.stat-badge span { color: #58a6ff; font-weight: 600; font-size: 1rem; display: block; }

.msg-user {
    background: #1f2a3a; border: 1px solid #1d4ed8;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px; margin: 8px 0; color: #e6edf3; font-size: 0.95rem;
}
.msg-agent {
    background: #1a2332; border: 1px solid #30363d;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px; margin: 8px 0; color: #e6edf3; font-size: 0.95rem;
}
.msg-label {
    font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
}
.msg-label.user  { color: #58a6ff; }
.msg-label.agent { color: #bc8cff; }

.trace-step {
    background: #161b22; border-left: 3px solid #238636;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px; margin: 6px 0; font-size: 0.85rem;
}
.trace-tool { color: #f0883e; font-weight: 600; }
.trace-obs  { color: #8b949e; margin-top: 4px; }

.stTextInput > div > div > input {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 10px !important; color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 16px !important; font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 10px 24px !important;
    font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
hr { border-color: #30363d !important; }
[data-testid="stFileUploader"] {
    background: #161b22; border: 1px dashed #30363d;
    border-radius: 10px; padding: 10px;
}
.stSpinner > div { border-top-color: #58a6ff !important; }
[data-testid="stExpander"] {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 10px !important;
}
.source-card {
    background: #161b22; border: 1px solid #30363d;
    border-left: 3px solid #58a6ff; border-radius: 0 8px 8px 0;
    padding: 10px 14px; margin: 5px 0; font-size: 0.85rem;
}
.source-title { color: #e6edf3; font-weight: 600; margin-bottom: 2px; }
.source-url   { color: #58a6ff; font-size: 0.78rem; word-break: break-all; }
.source-snip  { color: #8b949e; margin-top: 4px; font-size: 0.80rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_file_hash(file_bytes: bytes) -> str:
    """MD5 of file content — used for duplicate detection."""
    return hashlib.md5(file_bytes).hexdigest()

def ingest_document(file_path: str, doc_name: str) -> int:
    """
    Synchronous ingestion pipeline:
        Load PDF → split into chunks → Gemini embeddings → update FAISS → save.

    If a FAISS index already exists on disk, loads it and adds the new vectors.
    Returns the number of chunks created.
    """
    # 1. Load PDF pages
    loader = PyPDFLoader(file_path)
    pages  = loader.load()

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    # 3. Attach source metadata to every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"]      = doc_name
        chunk.metadata["page"]        = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_index"] = i

    # 4. Embed and persist to FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists(FAISS_INDEX_DIR):
        index = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        index.add_documents(chunks)
    else:
        index = FAISS.from_documents(chunks, embeddings)
    index.save_local(FAISS_INDEX_DIR)
    return len(chunks)

# ── Web Search Tool with Evidence Capture ──────────────────────────────────────
class WebSearchTool:
    """
    Wraps DDGS to give the ReAct agent plain-text search results while
    side-storing structured {title, url, snippet} evidence for display.

    Evidence is captured at the tool-result level — the LLM never sees or
    generates URLs; they come directly from the DDGS API response.
    """

    def reset(self) -> None:
        """Clear evidence before each new agent invocation."""
        self._evidence: list[dict] = []
        self._seen_urls: set[str]  = set()

    def __init__(self) -> None:
        self.reset()

    @property
    def evidence(self) -> list[dict]:
        return self._evidence

    def run(self, query: str) -> str:
        with _DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=5))
        text_parts = []
        for item in raw:
            url = item.get("href", "")
            # Deduplicate across multiple calls in the same agent run
            if url and url not in self._seen_urls:
                self._evidence.append({
                    "title":   item.get("title", ""),
                    "url":     url,
                    "snippet": item.get("body",  ""),
                })
                self._seen_urls.add(url)
            text_parts.append(f"{item.get('title', '')}: {item.get('body', '')}")
        return "\n\n".join(text_parts)

# ── Agent builder ──────────────────────────────────────────────────────────────
def build_agent(memory: ConversationBufferWindowMemory):
    """
    Load the persisted FAISS index and construct a fresh AgentExecutor.

    Called once when the first document is ready, and again whenever a new
    document is added (to reload FAISS with the updated index).
    The memory object is passed in so conversations survive rebuilds.

    Returns (AgentExecutor, WebSearchTool) or (None, None) if no FAISS index.
    """
    if not os.path.exists(FAISS_INDEX_DIR):
        return None, None

    # Load the FAISS index from disk
    embeddings   = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # ── Tool 1: Document Retrieval ─────────────────────────────────────────────
    def retrieve_documents(query: str) -> str:
        docs  = retriever.invoke(query)
        parts = []
        for doc in docs:
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[{src} | page {page}]\n{doc.page_content}")
        return "\n\n".join(parts)

    document_retrieval_tool = Tool(
        name        = "document_retrieval",
        func        = retrieve_documents,
        description = (
            "Search and retrieve information from uploaded documents. "
            "Returns relevant text chunks with source file and page number. "
            "Use this for any question about the uploaded PDFs."
        ),
    )

    # ── Tool 2: Calculator ─────────────────────────────────────────────────────
    calculator_tool             = PythonREPLTool()
    calculator_tool.name        = "calculator"
    calculator_tool.description = (
        "Evaluate mathematical expressions using Python. "
        "Input must be a valid Python expression, e.g. '0.15 * 4500000'."
    )

    # ── Tool 3: Web Search (with evidence capture) ─────────────────────────────
    # WebSearchTool.run() returns plain text to the LLM and simultaneously
    # populates WebSearchTool.evidence with {title, url, snippet} dicts.
    web_searcher    = WebSearchTool()
    web_search_tool = Tool(
        name        = "web_search",
        func        = web_searcher.run,
        description = (
            "Search the public web using DuckDuckGo. Use for current events, "
            "general knowledge, or anything not found in the uploaded documents."
        ),
    )

    tools = [document_retrieval_tool, calculator_tool, web_search_tool]

    # ── ReAct Agent with conversational memory ─────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model          = "gemini-3.6-flash",
        google_api_key = GOOGLE_API_KEY,
        temperature    = 0,
    )
    # hwchase17/react-chat includes {chat_history} — required for memory to work
    react_prompt = hub.pull("hwchase17/react-chat")
    agent        = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    executor = AgentExecutor(
        agent                     = agent,
        tools                     = tools,
        memory                    = memory,
        verbose                   = False,
        handle_parsing_errors     = (
            "Check your output format. You must use "
            "Thought/Action/Action Input or Thought/Final Answer."
        ),
        max_iterations            = 10,
        early_stopping_method     = "generate",
        return_intermediate_steps = True,
    )
    return executor, web_searcher

# ── Session state ──────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key      = "chat_history",
        k               = 5,           # last 5 exchanges sent to LLM
        return_messages = True,
        output_key      = "output",    # needed with return_intermediate_steps
    )
if "agent_executor"    not in st.session_state:
    st.session_state.agent_executor    = None
if "web_searcher"      not in st.session_state:
    st.session_state.web_searcher      = None
if "chat_history"      not in st.session_state:
    st.session_state.chat_history      = []
if "processed_hashes"  not in st.session_state:
    st.session_state.processed_hashes  = set()   # MD5s of ingested files
if "doc_names"         not in st.session_state:
    st.session_state.doc_names         = {}       # hash → filename

# If the app restarts but faiss_store/ already exists, rebuild the agent once
if st.session_state.agent_executor is None and os.path.exists(FAISS_INDEX_DIR):
    executor, searcher = build_agent(st.session_state.memory)
    st.session_state.agent_executor = executor
    st.session_state.web_searcher   = searcher

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drag & drop or browse",
        type                  = ["pdf"],
        accept_multiple_files = True,
        label_visibility      = "collapsed",
    )

    if uploaded_files:
        new_files = []
        for uf in uploaded_files:
            file_bytes = uf.read()
            file_hash  = get_file_hash(file_bytes)
            if file_hash not in st.session_state.processed_hashes:
                new_files.append((uf.name, file_hash, file_bytes))

        if new_files:
            for doc_name, file_hash, file_bytes in new_files:
                save_path = os.path.join(UPLOADS_DIR, f"{file_hash}.pdf")
                with open(save_path, "wb") as fh:
                    fh.write(file_bytes)

                with st.spinner(f"⚙️ Processing **{doc_name}** …"):
                    n_chunks = ingest_document(save_path, doc_name)

                st.session_state.processed_hashes.add(file_hash)
                st.session_state.doc_names[file_hash] = doc_name
                st.success(f"✅ {doc_name} — {n_chunks} chunks indexed")

            # Rebuild the agent to pick up newly added vectors
            with st.spinner("🔄 Updating knowledge base …"):
                executor, searcher = build_agent(st.session_state.memory)
                st.session_state.agent_executor = executor
                st.session_state.web_searcher   = searcher
            st.rerun()

    # Document list
    if st.session_state.doc_names:
        st.markdown("**Indexed Documents:**")
        for name in st.session_state.doc_names.values():
            st.markdown(f"✅ `{name}`")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#8b949e; line-height:1.8;">
    <b style="color:#e6edf3;">Model</b>&nbsp;&nbsp; gemini-3.6-flash<br>
    <b style="color:#e6edf3;">Embeddings</b>&nbsp;&nbsp; gemini-embedding-001<br>
    <b style="color:#e6edf3;">Memory</b>&nbsp;&nbsp; last 5 exchanges<br>
    <b style="color:#e6edf3;">Chunk size</b>&nbsp;&nbsp; {CHUNK_SIZE} chars<br>
    <b style="color:#e6edf3;">Overlap</b>&nbsp;&nbsp; {CHUNK_OVERLAP} chars<br>
    <b style="color:#e6edf3;">Top-k chunks</b>&nbsp;&nbsp; 4
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tools")
    st.markdown("""
    <div style="font-size:0.82rem; color:#8b949e; line-height:2;">
    📚 <b style="color:#f0883e;">document_retrieval</b><br>
    &nbsp;&nbsp;&nbsp;Semantic search across all PDFs<br>
    🔢 <b style="color:#f0883e;">calculator</b><br>
    &nbsp;&nbsp;&nbsp;Python math expressions<br>
    🌐 <b style="color:#f0883e;">web_search</b><br>
    &nbsp;&nbsp;&nbsp;DuckDuckGo public web search
    </div>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="hero-title">🧠 DocuSense AI</p>
<p class="hero-sub">Agentic Enterprise Knowledge Assistant &nbsp;·&nbsp; ReAct · Gemini · FAISS · Memory</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Main content ───────────────────────────────────────────────────────────────
TOOL_EMOJI = {"document_retrieval": "📚", "calculator": "🔢", "web_search": "🌐"}

if st.session_state.agent_executor:
    n_docs = len(st.session_state.doc_names)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-badge"><span>{n_docs}</span>Document(s) Indexed</div>
        <div class="stat-badge"><span>FAISS</span>Vector Store</div>
        <div class="stat-badge"><span>5-turn</span>Chat Memory</div>
        <div class="stat-badge"><span>ReAct</span>Agent Mode</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat history ───────────────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label user">You</div>
            {entry['q']}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View Agent Reasoning Trace"):
            if entry["trace"]:
                for action, observation in entry["trace"]:
                    emoji = TOOL_EMOJI.get(action.tool, "🔧")
                    st.markdown(f"""
                    <div class="trace-step">
                        <div class="trace-tool">{emoji} Tool: {action.tool}</div>
                        <div style="color:#58a6ff; font-size:0.82rem; margin:3px 0;">
                            Input: {action.tool_input}
                        </div>
                        <div class="trace-obs">
                            Observation: {str(observation)[:400]}{"..." if len(str(observation)) > 400 else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("Agent answered directly without calling any tool.")

        st.markdown(f"""
        <div class="msg-agent">
            <div class="msg-label agent">DocuSense AI</div>
            {entry['a']}
        </div>
        """, unsafe_allow_html=True)

        # ── Web Sources (only when web_search was used) ────────────────────────
        sources = entry.get("sources", [])
        if sources:
            with st.expander(f"🌐 Web Sources ({len(sources)})"):
                for i, src in enumerate(sources, 1):
                    domain = src["url"].split("/")[2] if src["url"] else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">{i}. {src['title']}</div>
                        <div class="source-url">🔗 {domain}</div>
                        <div class="source-snip">{src['snippet'][:220]}{'...' if len(src['snippet']) > 220 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"[Open source ↗]({src['url']})")

    # ── Question input ─────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input(
            "Ask a question",
            placeholder      = 'e.g. "What was the revenue and what is 15% of it?"',
            label_visibility = "collapsed",
            key              = "question_input",
        )
    with col2:
        ask_btn = st.button("Ask ✦", use_container_width=True)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()

    # ── Run the agent ──────────────────────────────────────────────────────────
    if ask_btn and user_query.strip():
        # Reset evidence before each query so sources belong to this response only
        if st.session_state.web_searcher:
            st.session_state.web_searcher.reset()

        with st.spinner("🤖 Agent is reasoning …"):
            result = st.session_state.agent_executor.invoke({"input": user_query})

        # Evidence is populated by WebSearchTool.run() during the agent run
        sources = (
            list(st.session_state.web_searcher.evidence)
            if st.session_state.web_searcher else []
        )

        st.session_state.chat_history.append({
            "q":       user_query,
            "a":       result["output"],
            "trace":   result.get("intermediate_steps", []),
            "sources": sources,   # [{title, url, snippet}] — empty if web_search unused
        })
        st.rerun()

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color:#8b949e;">
        <div style="font-size: 3.5rem; margin-bottom: 16px;">📂</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 8px;">
            Upload PDFs to get started
        </div>
        <div style="font-size: 0.9rem; max-width: 500px; margin: 0 auto; line-height: 1.7;">
            Upload one or more PDFs using the sidebar. Each document is processed
            immediately — chunked, embedded with Gemini, and stored in FAISS.
            Once done, the ReAct agent can retrieve facts, perform calculations,
            and search the web.
        </div>
        <br>
        <div style="font-size: 0.82rem; color: #58a6ff;">
            ← Use the sidebar to upload your PDFs
        </div>
    </div>
    """, unsafe_allow_html=True)
