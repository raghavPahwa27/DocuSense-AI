from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from .config import DocuSenseError
from .rag import configured_retriever
from .tools import build_tools
from datetime import datetime
from zoneinfo import ZoneInfo
import re
from typing import Optional

_INDIA_TZ = ZoneInfo("Asia/Kolkata")


def build_chat_model(settings):
    if settings.llm_provider == 'groq':
        return ChatGroq(model=settings.chat_model, api_key=settings.groq_api_key,
                        temperature=0, timeout=30, max_retries=1)
    return ChatGoogleGenerativeAI(model=settings.chat_model,
                                 google_api_key=settings.google_api_key, temperature=0)


def agent_error_message(error, provider, model):
    status = getattr(error, 'status_code', None)
    if status == 404:
        return f'{provider.title()} could not find or grant access to chat model {model}. Select an available model and retry.'
    if status in (401, 403):
        return f'{provider.title()} rejected access. Check the configured chat API key and its permissions.'
    if status == 429:
        return f'{provider.title()} rate limit or quota reached. Wait for the limit to reset or check your account quota.'
    if type(error).__name__ in ('APIConnectionError', 'APITimeoutError', 'TimeoutError'):
        return f'Cannot reach {provider.title()}. Check connectivity and retry.'
    if status == 400:
        return f'{provider.title()} rejected the chat request. The model may not support the request settings or the context may exceed its limit.'
    return 'Agent request failed. Please retry; if it persists, check provider availability and application configuration.'


def to_react_text(message):
    """Translate GPT-OSS native calls into the existing ReAct action format."""
    calls = getattr(message, 'tool_calls', [])
    if calls:
        if len(calls) != 1:
            raise DocuSenseError('The chat model returned multiple actions; retry with one request at a time.')
        call = calls[0]
        if call['name'] not in {'document_retrieval', 'calculator', 'web_search'}:
            raise DocuSenseError('The chat model requested an unsupported tool.')
        args = call.get('args', {})
        if not isinstance(args, dict) or len(args) != 1 or not isinstance(next(iter(args.values())), str):
            raise DocuSenseError('The chat model returned an invalid tool input.')
        return f"Action: {call['name']}\nAction Input: {next(iter(args.values()))}"
    text = str(message.content)
    if not text.strip():
        raise DocuSenseError('The chat model returned an empty response. Retry the request.')
    if 'Action:' in text or 'Final Answer:' in text:
        return text
    return 'Final Answer: ' + text

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


def ask_agent(executor, question: str):
    """Add fresh local time to every question; handle date-only requests directly."""
    payload = {'input': question, 'current_datetime': _now_context()}
    direct = _simple_date_answer(question)
    if direct is not None:
        executor.memory.save_context({'input': question}, {'output': direct})
        return {'input': question, 'output': direct, 'intermediate_steps': []}
    return executor.invoke(payload)

def build_agent(memory, store, settings):
    retriever = configured_retriever(store, settings)
    # The dynamic prompt variable is an internal input, not conversation text.
    # Pin memory to the user's actual question so LangChain does not treat the
    # timestamp as a second user input key.
    if hasattr(memory, "input_key"):
        memory.input_key = "input"
    tools, evidence = build_tools(retriever)

    # ── ReAct Agent with conversational memory ─────────────────────────────────
    llm = build_chat_model(settings)
    if settings.llm_provider == 'groq' and settings.chat_model.startswith('openai/gpt-oss-'):
        from langchain_core.runnables import RunnableLambda
        # GPT-OSS may emit native calls even with a text ReAct prompt. Declare
        # our existing tools and translate calls; AgentExecutor still runs them.
        llm = llm.bind_tools(tools, parallel_tool_calls=False) | RunnableLambda(to_react_text)
    # hwchase17/react-chat includes {chat_history} — required for memory to work
    react_prompt = hub.pull("hwchase17/react-chat")
    react_prompt.template += """

Trusted current date/time (Asia/Kolkata): {current_datetime}
Answer simple questions directly when the answer is in this context. If the user explicitly asks you to search the web or verify a website, call web_search before answering, even if you think you know the answer; date and weekday questions are the exception. For today's date or weekday, use the trusted current date/time above and do not call web_search. Use existing tools only when genuinely needed. Never fabricate facts, sources, search results, calculations, or document contents. If evidence is insufficient, explicitly say that the answer cannot be verified. For document questions, distinguish retrieved document information from general knowledge and claim a document fact only when the retrieved context supports it. Treat [WEB_SEARCH_NO_EVIDENCE] and [WEB_SEARCH_UNAVAILABLE] as absence of evidence, never as factual evidence.
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
    return executor, evidence

