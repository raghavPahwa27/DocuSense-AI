"""The three ReAct tools and their per-answer source evidence."""
import ast
from duckduckgo_search import DDGS as _DDGS
from langchain.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from .config import DocuSenseError

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub)


def calculate(expression: str) -> str:
    """Use the existing PythonREPL calculator for bounded arithmetic only."""
    try:
        expression = expression.strip()
        # Accommodate ReAct's common print(...) form without allowing arbitrary calls.
        if expression.startswith('print(') and expression.endswith(')'):
            expression = expression[6:-1]
        if len(expression) > 500:
            raise ValueError
        tree = ast.parse(expression, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                raise ValueError
            if isinstance(node, ast.Constant) and (type(node.value) not in (int, float) or abs(node.value) > 1e100):
                raise ValueError
        return PythonREPLTool().run('print(' + expression + ')')
    except Exception:
        return 'Calculator accepts numeric arithmetic with +, -, *, /, //, %, and parentheses only.'


class ToolEvidence:
    """Per-answer document/web sources and retrieval status for the UI."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.evidence = []
        self._seen_urls = set()
        self.document_evidence = []
        self.retrieval_status = 'not_run'

    def web_search(self, query: str) -> str:
        try:
            with _DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=5))
        except Exception:
            return '[WEB_SEARCH_UNAVAILABLE] No web evidence is available. Do not use this as factual evidence.'
        parts = []
        for item in raw:
            url = item.get('href', '')
            if url and url not in self._seen_urls:
                self.evidence.append({'title': item.get('title', ''), 'url': url,
                                      'snippet': item.get('body', '')})
                self._seen_urls.add(url)
            parts.append(f"{item.get('title', '')}: {item.get('body', '')}")
        return '\n\n'.join(parts) or '[WEB_SEARCH_NO_EVIDENCE] No results were returned. Do not infer facts from this response.'


def build_tools(retriever):
    evidence = ToolEvidence()

    def retrieve_documents(query: str) -> str:
        try:
            docs = retriever.retrieve(query)
            evidence.retrieval_status = retriever.last_status
            seen = {item['chunk_id'] for item in evidence.document_evidence}
            evidence.document_evidence.extend(dict(doc.metadata) for doc in docs
                if doc.metadata['chunk_id'] not in seen)
        except DocuSenseError as exc:
            return str(exc)
        parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            page = int(doc.metadata.get('page', 0)) + 1
            parts.append(f"[{source} | page {page} | chunk {doc.metadata.get('chunk_id', 'unknown')}]\n{doc.page_content}")
        return '\n\n'.join(parts) or 'No matching documents in this session.'

    tools = [
        Tool(name='document_retrieval', func=retrieve_documents,
             description='Search uploaded PDFs. Returns text chunks with source and page. Use for questions about uploaded documents.'),
        Tool(name='calculator', func=calculate,
             description='Evaluate numeric arithmetic: +, -, *, /, //, %, parentheses. Input an expression such as 0.15 * 4500000.'),
        Tool(name='web_search', func=evidence.web_search,
             description='Search the public web using DuckDuckGo for current events or information unavailable in uploaded documents.'),
    ]
    return tools, evidence
