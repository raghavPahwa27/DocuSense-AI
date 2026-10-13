from .llm import build_chat_model
from langchain.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from .calculator import calculate
from duckduckgo_search import DDGS as _DDGS
from langchain.memory import ConversationBufferWindowMemory
from .config import DocuSenseError
from .retrieval import configured_retriever
from datetime import datetime
from zoneinfo import ZoneInfo
import re
from typing import Optional

_INDIA_TZ = ZoneInfo("Asia/Kolkata")

_AGENT_GROUNDING_INSTRUCTIONS = """
Before answering, identify every distinct question or sub-question in the user's
message and answer each one. For multi-part questions, use numbered points or
clear sections in the same order so no part is silently omitted. Use
document_retrieval for parts about uploaded documents or applications; answer
general conceptual parts directly when sufficiently confident. If one part
cannot be answered reliably, answer the remaining parts and explicitly state
what evidence or information is missing for that part. For a date or weekday
part, use the trusted current date/time above and do not call web_search.
"""


def _now_context() -> str:
    now = datetime.now(_INDIA_TZ)
    return now.strftime("%A, %B %-d, %Y at %-I:%M:%S %p IST")


def _simple_date_answer(question: str) -> Optional[str]:
    normalized = question.lower().strip().rstrip("?.!")
    date_only = re.fullmatch(
        r"(?:what(?:\s+is|'s)\s+)?today(?:'s|s)?\s+(?:day|date)",
        normalized,
    )
    if date_only:
        now = datetime.now(_INDIA_TZ)
        if "date" in normalized and "day" not in normalized:
            return now.strftime("Today is %B %-d, %Y.")
        return now.strftime("Today is %A, %B %-d, %Y.")
    if normalized in {"what day is it", "what is the day today"}:
        return datetime.now(_INDIA_TZ).strftime("Today is %A.")
    return None


class DynamicAgentExecutor:
    """Keep the existing AgentExecutor while adding per-request trusted time."""
    def __init__(self, executor):
        self._executor = executor
        self.memory = executor.memory

    def invoke(self, inputs, config=None, **kwargs):
        payload = dict(inputs)
        payload["current_datetime"] = _now_context()
        direct = _simple_date_answer(str(payload.get("input", "")))
        if direct is not None:
            self.memory.save_context({"input": payload["input"]}, {"output": direct})
            return {"input": payload["input"], "output": direct, "intermediate_steps": []}
        return self._executor.invoke(payload, config=config, **kwargs)

    def __getattr__(self, name):
        return getattr(self._executor, name)

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
        self.document_evidence = []
        self.retrieval_status = "not_run"

    def __init__(self) -> None:
        self.reset()

    @property
    def evidence(self) -> list[dict]:
        return self._evidence

    def run(self, query: str) -> str:
        try:
            with _DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=5))
        except Exception:
            return "[WEB_SEARCH_UNAVAILABLE] No web evidence is available. Do not use this as factual evidence."
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
        return "\n\n".join(text_parts) or "[WEB_SEARCH_NO_EVIDENCE] No results were returned. Do not infer facts from this response."

def build_agent(memory, store, settings):
    retriever = configured_retriever(store, settings)
    # The dynamic prompt variable is an internal input, not conversation text.
    # Pin memory to the user's actual question so LangChain does not treat the
    # timestamp as a second user input key.
    if hasattr(memory, "input_key"):
        memory.input_key = "input"
    # ── Tool 1: Document Retrieval ─────────────────────────────────────────────
    def retrieve_documents(query: str) -> str:
        try:
            docs = retriever.retrieve(query)
            web_searcher.retrieval_status = retriever.last_status
            seen = {d["chunk_id"] for d in web_searcher.document_evidence}
            web_searcher.document_evidence.extend(dict(d.metadata) for d in docs if d.metadata["chunk_id"] not in seen)
        except DocuSenseError as exc:
            return str(exc)
        parts = []
        for doc in docs:
            src  = doc.metadata.get("source", "unknown")
            page = int(doc.metadata.get("page", 0)) + 1
            parts.append(f"[{src} | page {page} | chunk {doc.metadata.get('chunk_id', 'unknown')}]\n{doc.page_content}")
        return "\n\n".join(parts) or "No matching documents in this session."

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
    calculator_tool = Tool(name="calculator", func=calculate, description=(
        "Evaluate numeric arithmetic using Python: +, -, *, /, //, %, parentheses. "
        "Input an expression such as 0.15 * 4500000."
    ))

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
    llm = build_chat_model(settings)
    if settings.llm_provider == 'groq' and settings.chat_model.startswith('openai/gpt-oss-'):
        from langchain_core.runnables import RunnableLambda
        from .react_adapter import to_react_text
        # GPT-OSS may emit native calls even with a text ReAct prompt. Declare
        # our existing tools and translate calls; AgentExecutor still runs them.
        llm = llm.bind_tools(tools, parallel_tool_calls=False) | RunnableLambda(to_react_text)
    # hwchase17/react-chat includes {chat_history} — required for memory to work
    react_prompt = hub.pull("hwchase17/react-chat")
    react_prompt.template += """

Trusted current date/time (Asia/Kolkata): {current_datetime}
Answer simple questions directly when the answer is in this context. For today's date or weekday, use the trusted current date/time above and do not call web_search. Use existing tools only when genuinely needed. Never fabricate facts, sources, search results, calculations, or document contents. If evidence is insufficient, explicitly say that the answer cannot be verified. For document questions, distinguish retrieved document information from general knowledge and claim a document fact only when the retrieved context supports it. Treat [WEB_SEARCH_NO_EVIDENCE] and [WEB_SEARCH_UNAVAILABLE] as absence of evidence, never as factual evidence.
""" + _AGENT_GROUNDING_INSTRUCTIONS
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
        early_stopping_method     = "force",
        return_intermediate_steps = True,
    )
    return DynamicAgentExecutor(executor), web_searcher

