"""Actionable provider errors without exposing response bodies or credentials."""


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
