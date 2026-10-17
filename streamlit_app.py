import os
import uuid
from html import escape
from urllib.parse import urlparse
from pathlib import Path
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from docusense.config import Settings, DocuSenseError, CHUNK_SIZE, CHUNK_OVERLAP, ROOT
from docusense.document_manager import connect_document_manager
from docusense.rag import digest
from docusense.agent import build_agent, ask_agent, agent_error_message

UPLOADS_DIR = ROOT / 'uploads' / 'sessions'

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
if "tool_evidence"     not in st.session_state:
    st.session_state.tool_evidence     = None
    # Rebuild a pre-refactor agent after Streamlit hot reloads this code.
    st.session_state.agent_executor    = None
if "chat_history"      not in st.session_state:
    st.session_state.chat_history      = []
if "processed_hashes"  not in st.session_state:
    st.session_state.processed_hashes  = set()   # SHA-256 IDs of ingested files
if "doc_names"         not in st.session_state:
    st.session_state.doc_names         = {}       # hash → filename

# Never use a process-wide owner or a cached cross-session agent/store.
if "owner_id" not in st.session_state:
    st.session_state.owner_id = uuid.uuid4().hex
if "store" not in st.session_state:
    st.session_state.store = None
settings = None
try:
    # Streamlit 1.38 renders an error before raising on a missing secrets file.
    # Its optional loader avoids that UI error for valid .env-only setups.
    secrets = dict(st.secrets) if st.secrets.load_if_toml_exists() else {}
    settings = Settings.load(secrets)
except DocuSenseError as exc:
    st.warning(str(exc))

# Rebuild cached agents when chat configuration changes, preserving conversation memory.
if settings is not None:
    chat_configuration = (settings.llm_provider, settings.chat_model,
                          digest((settings.groq_api_key if settings.llm_provider == "groq" else settings.google_api_key).encode()))
    if st.session_state.get("chat_configuration") != chat_configuration:
        st.session_state.agent_executor = None
        st.session_state.tool_evidence = None
        st.session_state.chat_configuration = chat_configuration

def session_store():
    if settings is None:
        raise DocuSenseError("Configure chat, embedding, and Pinecone credentials first.")
    if st.session_state.store is None:
        st.session_state.store = connect_document_manager(settings, st.session_state.owner_id)
    return st.session_state.store

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
            file_bytes = uf.getvalue()
            file_hash  = digest(file_bytes)
            if file_hash not in st.session_state.processed_hashes:
                new_files.append((uf.name, file_hash, file_bytes))

        if new_files:
            for doc_name, file_hash, file_bytes in new_files:
                try:
                    owner_dir = UPLOADS_DIR / st.session_state.owner_id
                    owner_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                    save_path = owner_dir / f"{file_hash}.pdf"
                    save_path.write_bytes(file_bytes)
                    save_path.chmod(0o600)
                    with st.spinner(f"⚙️ Processing **{doc_name}** …"):
                        result = session_store().ingest_pdf(save_path, doc_name)
                    st.session_state.processed_hashes.add(file_hash)
                    st.session_state.doc_names[file_hash] = doc_name
                    st.success(f"✅ {doc_name} — {result.chunk_count} chunks indexed")
                except DocuSenseError as exc:
                    st.error(str(exc))
                except OSError:
                    st.error("Cannot save the upload. Check local storage permissions.")

    if st.session_state.doc_names and st.session_state.agent_executor is None:
        try:
            with st.spinner("🔄 Updating knowledge base …"):
                executor, evidence = build_agent(st.session_state.memory, session_store(), settings)
                st.session_state.agent_executor = executor
                st.session_state.tool_evidence = evidence
        except Exception:
            st.error("Agent initialization failed. Check chat-provider access and LangChain Hub connectivity, then retry.")

    # Document mutations always use this session's store, never a user-supplied owner.
    if st.session_state.doc_names:
        with st.expander("Manage documents"):
            names = dict(st.session_state.doc_names)
            selected = st.selectbox("Document", list(names), key="managed_document",
                                    format_func=lambda key, names=names: names[key])
            if st.button("Re-index selected document"):
                try:
                    path = UPLOADS_DIR / st.session_state.owner_id / f"{selected}.pdf"
                    session_store().ingest_pdf(path, st.session_state.doc_names[selected], reindex=True)
                    st.session_state.memory.clear()
                    st.session_state.chat_history = []
                    st.success("Document re-indexed and verified. Conversation cleared to remove stale context.")
                except DocuSenseError as exc:
                    st.error(str(exc))
            if st.button("Delete selected document from Pinecone"):
                # Remove old conversation context even if remote deletion needs a retry.
                st.session_state.memory.clear()
                st.session_state.chat_history = []
                try:
                    session_store().delete_document(selected)
                    del st.session_state.doc_names[selected]
                    # Keep processed_hashes so a still-selected upload is not re-ingested.
                    st.success("Document removed from retrieval. Original PDF remains on disk.")
                    st.rerun()
                except DocuSenseError as exc:
                    st.error(str(exc))

    # Document list
    if st.session_state.doc_names:
        st.markdown("**Indexed Documents:**")
        for name in st.session_state.doc_names.values():
            st.markdown(f"✅ `{name}`")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#8b949e; line-height:1.8;">
    <b style="color:#e6edf3;">Model</b>&nbsp;&nbsp; {escape(settings.chat_model) if settings else "not configured"}<br>
    <b style="color:#e6edf3;">Embeddings</b>&nbsp;&nbsp; gemini-embedding-001<br>
    <b style="color:#e6edf3;">Memory</b>&nbsp;&nbsp; last 5 exchanges<br>
    <b style="color:#e6edf3;">Chunk size</b>&nbsp;&nbsp; {CHUNK_SIZE} chars<br>
    <b style="color:#e6edf3;">Overlap</b>&nbsp;&nbsp; {CHUNK_OVERLAP} chars<br>
    <b style="color:#e6edf3;">Top-k chunks</b>&nbsp;&nbsp; {settings.top_k if settings else 4}<br>
    <b style="color:#e6edf3;">Reranking</b>&nbsp;&nbsp; {"enabled" if settings and settings.rerank_enabled else "disabled"}
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
<p class="hero-sub">Agentic Enterprise Knowledge Assistant &nbsp;·&nbsp; ReAct · Pinecone · Memory</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Main content ───────────────────────────────────────────────────────────────
TOOL_EMOJI = {"document_retrieval": "📚", "calculator": "🔢", "web_search": "🌐"}

if st.session_state.agent_executor:
    n_docs = len(st.session_state.doc_names)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-badge"><span>{n_docs}</span>Document(s) Indexed</div>
        <div class="stat-badge"><span>Pinecone</span>Vector Store</div>
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
            {escape(entry['q'])}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View Agent Reasoning Trace"):
            if entry["trace"]:
                for action, observation in entry["trace"]:
                    emoji = TOOL_EMOJI.get(action.tool, "🔧")
                    st.markdown(f"""
                    <div class="trace-step">
                        <div class="trace-tool">{emoji} Tool: {escape(action.tool)}</div>
                        <div style="color:#58a6ff; font-size:0.82rem; margin:3px 0;">
                            Input: {escape(str(action.tool_input))}
                        </div>
                        <div class="trace-obs">
                            Observation: {escape(str(observation)[:400])}{"..." if len(str(observation)) > 400 else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("Agent answered directly without calling any tool.")

        st.markdown(f"""
        <div class="msg-agent">
            <div class="msg-label agent">DocuSense AI</div>
            {escape(str(entry['a']))}
        </div>
        """, unsafe_allow_html=True)

        document_sources = entry.get("document_sources", [])
        if document_sources:
            with st.expander(f"📚 Document Sources ({len(document_sources)})"):
                for src in document_sources:
                    st.text(f"{src['source']} | page {int(src['page']) + 1} | chunk {src['chunk_id']}")
        if entry.get("retrieval_status", "").startswith("fallback:"):
            st.caption("Reranking was unavailable; this answer used vector-only retrieval.")

        # ── Web Sources (only when web_search was used) ────────────────────────
        sources = entry.get("sources", [])
        if sources:
            with st.expander(f"🌐 Web Sources ({len(sources)})"):
                for i, src in enumerate(sources, 1):
                    domain = escape(urlparse(src["url"]).netloc)
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">{i}. {escape(src['title'])}</div>
                        <div class="source-url">🔗 {domain}</div>
                        <div class="source-snip">{escape(src['snippet'][:220])}{'...' if len(src['snippet']) > 220 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if urlparse(src["url"]).scheme in {"http", "https"}:
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
        if st.session_state.tool_evidence:
            st.session_state.tool_evidence.reset()

        try:
            with st.spinner("🤖 Agent is reasoning …"):
                result = ask_agent(st.session_state.agent_executor, user_query)
        except Exception as exc:
            st.error(agent_error_message(exc, settings.llm_provider, settings.chat_model))
            st.stop()

        # Evidence is populated by the existing tools during this agent run.
        sources = (
            list(st.session_state.tool_evidence.evidence)
            if st.session_state.tool_evidence else []
        )

        st.session_state.chat_history.append({
            "q":       user_query,
            "a":       result["output"],
            "trace":   result.get("intermediate_steps", []),
            "document_sources": list(st.session_state.tool_evidence.document_evidence),
            "retrieval_status": st.session_state.tool_evidence.retrieval_status,
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
            immediately — chunked, embedded with Gemini, and stored in Pinecone.
            Once done, the ReAct agent can retrieve facts, perform calculations,
            and search the web.
        </div>
        <br>
        <div style="font-size: 0.82rem; color: #58a6ff;">
            ← Use the sidebar to upload your PDFs
        </div>
    </div>
    """, unsafe_allow_html=True)
