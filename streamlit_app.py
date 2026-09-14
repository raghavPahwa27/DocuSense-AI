import os
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool

# ── Config ─────────────────────────────────────────────────────────────────────
# Use absolute path so load_dotenv() works regardless of Streamlit's cwd
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 150

# ── Page Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "DocuSense AI",
    page_icon  = "🧠",
    layout     = "wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark base */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Main header */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #ff7b72 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* Stat badges */
.stat-row { display: flex; gap: 12px; margin-top: 1rem; flex-wrap: wrap; }
.stat-badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #8b949e;
}
.stat-badge span { color: #58a6ff; font-weight: 600; font-size: 1rem; display: block; }

/* Chat messages */
.msg-user {
    background: #1f2a3a;
    border: 1px solid #1d4ed8;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e6edf3;
    font-size: 0.95rem;
}
.msg-agent {
    background: #1a2332;
    border: 1px solid #30363d;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e6edf3;
    font-size: 0.95rem;
}
.msg-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.msg-label.user { color: #58a6ff; }
.msg-label.agent { color: #bc8cff; }

/* Trace step */
.trace-step {
    background: #161b22;
    border-left: 3px solid #238636;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
}
.trace-tool { color: #f0883e; font-weight: 600; }
.trace-obs { color: #8b949e; margin-top: 4px; }

/* Input area */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Divider */
hr { border-color: #30363d !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 10px;
    padding: 10px;
}

/* Spinner */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* Success / info boxes */
.stSuccess { background: #0d2a1e !important; border: 1px solid #238636 !important; border-radius: 8px !important; }
.stInfo    { background: #0d1f36 !important; border: 1px solid #1d4ed8 !important; border-radius: 8px !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}

/* Sidebar upload status */
.upload-status {
    background: #0d2a1e;
    border: 1px solid #238636;
    border-radius: 8px;
    padding: 12px;
    margin-top: 12px;
    font-size: 0.85rem;
    color: #3fb950;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#8b949e; line-height:1.8;">
    <b style="color:#e6edf3;">Model</b>&nbsp;&nbsp; gemini-3.6-flash<br>
    <b style="color:#e6edf3;">Embeddings</b>&nbsp;&nbsp; gemini-embedding-001<br>
    <b style="color:#e6edf3;">Chunk size</b>&nbsp;&nbsp; {CHUNK_SIZE} chars<br>
    <b style="color:#e6edf3;">Overlap</b>&nbsp;&nbsp; {CHUNK_OVERLAP} chars<br>
    <b style="color:#e6edf3;">Top-k chunks</b>&nbsp;&nbsp; 4
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tools Available")
    st.markdown("""
    <div style="font-size:0.82rem; color:#8b949e; line-height:2;">
    📚 <b style="color:#f0883e;">document_retrieval</b><br>
    &nbsp;&nbsp;&nbsp;Semantic search over PDF<br>
    🔢 <b style="color:#f0883e;">calculator</b><br>
    &nbsp;&nbsp;&nbsp;Python math expressions
    </div>
    """, unsafe_allow_html=True)

# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<p class="hero-title">🧠 DocuSense AI</p>
<p class="hero-sub">Agentic Enterprise Knowledge Assistant &nbsp;·&nbsp; ReAct · Gemini · FAISS</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Session state init ─────────────────────────────────────────────────────────
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "indexed_file"   not in st.session_state:
    st.session_state.indexed_file   = None
if "page_count"     not in st.session_state:
    st.session_state.page_count     = 0
if "chunk_count"    not in st.session_state:
    st.session_state.chunk_count    = 0
if "chat_history"   not in st.session_state:
    st.session_state.chat_history   = []   # list of {"q": ..., "a": ..., "trace": [...]}

# ── Build agent when a new PDF is uploaded ─────────────────────────────────────
if uploaded_file and uploaded_file.name != st.session_state.indexed_file:

    with st.spinner(f"📥 Indexing **{uploaded_file.name}** — chunking → embedding → FAISS ..."):

        # Save upload to a temp file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Step 1 – Load
        loader    = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Step 2 – Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        # Step 3 – Gemini Embeddings → FAISS
        embeddings   = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever    = vector_store.as_retriever(search_kwargs={"k": 4})

        # Step 4 – Document Retrieval Tool
        def retrieve_documents(query: str) -> str:
            relevant_chunks = retriever.invoke(query)
            return "\n\n".join(doc.page_content for doc in relevant_chunks)

        document_retrieval_tool = Tool(
            name        = "document_retrieval",
            func        = retrieve_documents,
            description = (
                "Search and retrieve information from the uploaded enterprise document. "
                "Input should be a natural language query about the document contents."
            ),
        )

        # Step 5 – Calculator Tool
        calculator_tool             = PythonREPLTool()
        calculator_tool.name        = "calculator"
        calculator_tool.description = (
            "Perform mathematical calculations. "
            "Input must be a valid Python math expression, e.g. '0.15 * 4500000'."
        )

        tools = [document_retrieval_tool, calculator_tool]

        # Step 6 – ReAct Agent
        llm = ChatGoogleGenerativeAI(
            model          = "gemini-3.6-flash",
            google_api_key = GOOGLE_API_KEY,
            temperature    = 0,
        )
        react_prompt = hub.pull("hwchase17/react")
        agent        = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

        # Step 7 – Agent Executor
        # return_intermediate_steps=True lets us show the reasoning trace in the UI.
        # early_stopping_method="generate" forces a final answer if max_iterations hit.
        agent_executor = AgentExecutor(
            agent                     = agent,
            tools                     = tools,
            verbose                   = False,
            handle_parsing_errors     = "Check your output format. You must output Thought/Action/Action Input or Thought/Final Answer.",
            max_iterations            = 10,
            early_stopping_method     = "generate",
            return_intermediate_steps = True,
        )

        # Persist in session state
        st.session_state.agent_executor = agent_executor
        st.session_state.indexed_file   = uploaded_file.name
        st.session_state.page_count     = len(documents)
        st.session_state.chunk_count    = len(chunks)
        st.session_state.chat_history   = []   # reset chat on new upload

    st.success(f"✅ **{uploaded_file.name}** indexed successfully!")

# ── Stats row (shown once a file is indexed) ───────────────────────────────────
if st.session_state.agent_executor:
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-badge"><span>{st.session_state.indexed_file}</span>Active Document</div>
        <div class="stat-badge"><span>{st.session_state.page_count}</span>Pages Loaded</div>
        <div class="stat-badge"><span>{st.session_state.chunk_count}</span>Chunks Indexed</div>
        <div class="stat-badge"><span>FAISS</span>Vector Store</div>
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
                    tool_emoji = "📚" if action.tool == "document_retrieval" else "🔢"
                    st.markdown(f"""
                    <div class="trace-step">
                        <div class="trace-tool">{tool_emoji} Tool: {action.tool}</div>
                        <div style="color:#58a6ff; font-size:0.82rem; margin:3px 0;">Input: {action.tool_input}</div>
                        <div class="trace-obs">Observation: {str(observation)[:400]}{"..." if len(str(observation)) > 400 else ""}</div>
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

    # ── Question input ─────────────────────────────────────────────────────────
    st.markdown("---")

    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input(
            "Ask a question about your document",
            placeholder='e.g. "What was the total revenue and what is 15% of it?"',
            label_visibility="collapsed",
            key="question_input",
        )
    with col2:
        ask_btn = st.button("Ask ✦", use_container_width=True)

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Run the agent ──────────────────────────────────────────────────────────
    if ask_btn and user_query.strip():
        with st.spinner("🤖 Agent is reasoning ..."):
            result = st.session_state.agent_executor.invoke({"input": user_query})

        # Save to history
        st.session_state.chat_history.append({
            "q":     user_query,
            "a":     result["output"],
            "trace": result.get("intermediate_steps", []),
        })
        st.rerun()

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color:#8b949e;">
        <div style="font-size: 3.5rem; margin-bottom: 16px;">📂</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 8px;">
            Upload a PDF to get started
        </div>
        <div style="font-size: 0.9rem; max-width: 480px; margin: 0 auto; line-height: 1.7;">
            DocuSense AI will chunk your document, generate Gemini embeddings,
            build a FAISS vector index, and hand it all to a ReAct agent
            that can retrieve facts <em>and</em> perform calculations — in one loop.
        </div>
        <br>
        <div style="font-size: 0.82rem; color: #58a6ff;">
            ← Use the sidebar to upload your PDF
        </div>
    </div>
    """, unsafe_allow_html=True)
