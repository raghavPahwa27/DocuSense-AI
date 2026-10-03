import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

import numpy as np
import faiss as faiss_lib

from langchain_core.embeddings import Embeddings as _LCEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # required: gRPC transport reads env var, not kwarg
DATA_DIR       = "data/"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 150

# ── Step 1: Load Documents ─────────────────────────────────────────────────────
# PyPDFDirectoryLoader reads every PDF inside the data/ folder.
print("📂 Loading documents from data/ ...")
loader    = PyPDFDirectoryLoader(DATA_DIR)
documents = loader.load()
print(f"   Loaded {len(documents)} page(s).")

# ── Step 2: Split Documents into Chunks ────────────────────────────────────────
# Large documents are split into overlapping chunks so each chunk fits in the
# context window and overlapping preserves sentence continuity across boundaries.
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
)
chunks = splitter.split_documents(documents)
print(f"   Split into {len(chunks)} chunk(s).")

# ── Step 3: Gemini Embeddings + FAISS Vector Store (cosine similarity) ────────
# Each chunk is converted into a dense vector by Gemini's embedding model.
# Vectors are L2-normalised before being stored in an IndexFlatIP index so that
# inner-product search corresponds exactly to cosine similarity:
#   cosine(a, b) = a·b  when ‖a‖ = ‖b‖ = 1
print("🔢 Creating embeddings and building FAISS index (cosine / IndexFlatIP) ...")
_base_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

class _NormalisingEmbeddings(_LCEmbeddings):
    """
    LangChain Embeddings adapter that L2-normalises query embeddings.
    Document embeddings are normalised at build time inside _build_cosine_faiss.
    This class ensures query vectors are also on the unit sphere so that
    IndexFlatIP inner-product retrieval equals cosine similarity.
    Inherits from langchain_core.embeddings.Embeddings to satisfy the type
    contract; __call__ covers the legacy callable path inside LangChain's
    community FAISS implementation.
    """
    def embed_documents(self, texts: list) -> list:
        return _base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list:
        raw = _base_embeddings.embed_query(text)
        vec = np.array([raw], dtype=np.float32)
        faiss_lib.normalize_L2(vec)      # in-place unit-normalisation
        return vec[0].tolist()

    # Legacy callable path used by some LangChain FAISS internals
    def __call__(self, text: str) -> list:
        return self.embed_query(text)

    def __getattr__(self, name):
        return getattr(_base_embeddings, name)

embeddings = _NormalisingEmbeddings()

def _build_cosine_faiss(
    documents: list,
    embedding_model: _LCEmbeddings,
) -> FAISS:
    """
    Build a LangChain FAISS vector store backed by an IndexFlatIP index.

    All embeddings are L2-normalised before insertion so that FAISS inner-
    product search returns cosine-similarity scores rather than raw dot
    products.  The embedding model and chunking strategy are unchanged.
    """
    # Embed all chunks in one batch via the base model (raw vectors)
    texts    = [doc.page_content for doc in documents]
    raw_vecs = _base_embeddings.embed_documents(texts)

    # Normalise to unit length (float32 required by FAISS)
    vectors = np.array(raw_vecs, dtype=np.float32)
    faiss_lib.normalize_L2(vectors)        # in-place L2 normalisation

    # Build IndexFlatIP (cosine similarity via inner product on unit vectors)
    dim   = vectors.shape[1]
    index = faiss_lib.IndexFlatIP(dim)
    index.add(vectors)

    # Wrap in a LangChain FAISS store; pass the *normalising* embedding wrapper
    # so that future similarity_search() calls also normalise query vectors.
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    docstore = InMemoryDocstore(
        {str(i): doc for i, doc in enumerate(documents)}
    )
    return FAISS(
        embedding_function   = embedding_model,
        index                = index,
        docstore             = docstore,
        index_to_docstore_id = index_to_docstore_id,
    )

vector_store = _build_cosine_faiss(chunks, embeddings)
retriever    = vector_store.as_retriever(search_kwargs={"k": 4})
print("   FAISS index ready (IndexFlatIP / cosine).\n")

# ── Step 4: Document Retrieval Tool ────────────────────────────────────────────
# The agent calls this tool when it needs factual information from the documents.
# It runs a semantic similarity search and returns the top-k chunks as a string.
def retrieve_documents(query: str) -> str:
    relevant_chunks = retriever.invoke(query)
    # Join all retrieved chunks into one string the agent can read.
    return "\n\n".join(doc.page_content for doc in relevant_chunks)

document_retrieval_tool = Tool(
    name        = "document_retrieval",
    func        = retrieve_documents,
    description = (
        "Use this tool to search and retrieve information from the enterprise "
        "documents. Input should be a natural language query about the documents."
    ),
)

# ── Step 5: Calculator Tool ────────────────────────────────────────────────────
# The agent calls this tool for any mathematical computation.
# PythonREPLTool safely evaluates Python math expressions and returns the result.
calculator_tool = PythonREPLTool()
calculator_tool.name        = "calculator"
calculator_tool.description = (
    "Use this tool to perform mathematical calculations. "
    "Input must be a valid Python math expression, e.g. '0.15 * 4500000'."
)

# ── Step 6: Web Search Tool ────────────────────────────────────────────────────────────────
# The agent calls this tool when it needs current or external information
# that is not present inside the enterprise documents.
web_search_tool = DuckDuckGoSearchRun()
web_search_tool.name        = "web_search"
web_search_tool.description = (
    "Search the public web using DuckDuckGo. Use this for current events, "
    "publicly available information, or anything not found in the enterprise documents."
)

tools = [document_retrieval_tool, calculator_tool, web_search_tool]

# ── Step 6: ReAct Agent ────────────────────────────────────────────────────────
# The ReAct (Reason + Act) pattern lets the LLM alternate between:
#   Thought  -> what does the agent plan to do next?
#   Action   -> which tool should it call, and with what input?
#   Observation -> what did the tool return?
# This loop continues until the agent is confident it has the Final Answer.
llm = ChatGoogleGenerativeAI(
    model          = "gemini-3.6-flash",
    google_api_key = GOOGLE_API_KEY,
    temperature    = 0,          # deterministic answers
)

# Pull the standard ReAct prompt template from LangChain Hub.
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm     = llm,
    tools   = tools,
    prompt  = react_prompt,
)

# ── Step 7: Agent Executor ─────────────────────────────────────────────────────
# AgentExecutor manages the Agent -> Action -> Tool -> Observation -> Agent loop.
# verbose=True prints each Thought / Action / Observation so you can trace the
# agent's reasoning -- very useful for demos and interviews.
agent_executor = AgentExecutor(
    agent                 = agent,
    tools                 = tools,
    verbose               = True,
    handle_parsing_errors = "Check your output format. You must output Thought/Action/Action Input or Thought/Final Answer.",
    max_iterations        = 10,
    early_stopping_method = "generate",
)

# ── Step 8: Terminal Question-Answer Loop ──────────────────────────────────────
print("=" * 60)
print("  DocuSense AI – Agentic Enterprise Knowledge Assistant")
print("  Type your question and press Enter. Type 'exit' to quit.")
print("=" * 60)

while True:
    user_query = input("\nYou: ").strip()
    if user_query.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if not user_query:
        continue

    result = agent_executor.invoke({"input": user_query})
    print(f"\n🤖 Answer: {result['output']}")
