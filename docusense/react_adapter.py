"""Translate native single-tool responses into the existing ReAct text protocol."""
from .config import DocuSenseError


def to_react_text(message):
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
