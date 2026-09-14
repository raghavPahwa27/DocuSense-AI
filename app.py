import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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

# ── Step 3: Gemini Embeddings + FAISS Vector Store ─────────────────────────────
# Each chunk is converted into a dense vector by Gemini's embedding model.
# FAISS indexes these vectors so we can do fast similarity search at query time.
print("🔢 Creating embeddings and building FAISS index ...")
embeddings   = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)
vector_store = FAISS.from_documents(chunks, embeddings)
retriever    = vector_store.as_retriever(search_kwargs={"k": 4})
print("   FAISS index ready.\n")

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

tools = [document_retrieval_tool, calculator_tool]

# ── Step 6: ReAct Agent ────────────────────────────────────────────────────────
# The ReAct (Reason + Act) pattern lets the LLM alternate between:
#   Thought  -> what does the agent plan to do next?
#   Action   -> which tool should it call, and with what input?
#   Observation -> what did the tool return?
# This loop continues until the agent is confident it has the Final Answer.
llm = ChatGoogleGenerativeAI(
    model          = "gemini-1.5-flash",
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
    agent   = agent,
    tools   = tools,
    verbose = True,
    handle_parsing_errors = True,
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
