from dataclasses import replace
from unittest.mock import Mock
from docusense.config import Settings
from docusense.llm import build_chat_model


def test_groq_chat_uses_groq_key_not_embedding_key(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr('docusense.llm.ChatGroq', constructor)
    settings = Settings('embedding-key', 'pinecone-key', 'index', llm_provider='groq',
                        chat_model='llama-3.3-70b-versatile', groq_api_key='chat-key')
    build_chat_model(settings)
    assert constructor.call_args.kwargs['api_key'] == 'chat-key'
    assert constructor.call_args.kwargs['model'] == 'llama-3.3-70b-versatile'
    assert 'embedding-key' not in str(constructor.call_args)
    assert 'chat-key' not in repr(settings)


def test_gemini_chat_remains_available(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr('docusense.llm.ChatGoogleGenerativeAI', constructor)
    build_chat_model(Settings('embedding-key','pinecone-key','index'))
    assert constructor.call_args.kwargs['model'] == 'gemini-3.6-flash'
