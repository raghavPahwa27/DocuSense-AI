from docusense.agent import agent_error_message


def test_provider_messages_do_not_expose_exception_body():
    class ProviderError(Exception):
        status_code = 404
    err = ProviderError('secret-key and confidential document text')
    message = agent_error_message(err, 'groq', 'missing-model')
    assert 'missing-model' in message
    assert 'secret-key' not in message
    assert 'confidential' not in message
    err.status_code = 429
    assert 'quota' in agent_error_message(err, 'groq', 'model')
    err.status_code = 401
    assert 'permissions' in agent_error_message(err, 'groq', 'model')
