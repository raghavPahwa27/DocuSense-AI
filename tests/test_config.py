import pytest
from docusense.config import Settings, DocuSenseError


def test_secrets_fallback_and_environment_precedence(monkeypatch):
    monkeypatch.setattr('docusense.config.load_dotenv', lambda *a,**k: None)
    for name in ['GOOGLE_API_KEY','PINECONE_API_KEY','PINECONE_INDEX_NAME','RERANK_ENABLED','RETRIEVAL_CANDIDATES','RETRIEVAL_TOP_K']:
        monkeypatch.delenv(name, raising=False)
    values = {'GOOGLE_API_KEY':'g','PINECONE_API_KEY':'p','PINECONE_INDEX_NAME':'index'}
    assert Settings.load(values).index_name == 'index'
    monkeypatch.setenv('PINECONE_INDEX_NAME','environment-index')
    settings = Settings.load(values)
    assert settings.index_name == 'environment-index'
    assert 'google_api_key=' not in repr(settings)
    assert 'pinecone_api_key=' not in repr(settings)


def test_invalid_retrieval_config(monkeypatch):
    monkeypatch.setenv('RETRIEVAL_TOP_K','21')
    monkeypatch.setenv('RETRIEVAL_CANDIDATES','20')
    with pytest.raises(DocuSenseError, match='TOP_K'):
        Settings.load({'GOOGLE_API_KEY':'g','PINECONE_API_KEY':'p','PINECONE_INDEX_NAME':'index'})


def test_groq_settings_from_secrets(monkeypatch):
    monkeypatch.setattr('docusense.config.load_dotenv', lambda *a, **k: None)
    for name in ['GOOGLE_API_KEY','PINECONE_API_KEY','PINECONE_INDEX_NAME','LLM_PROVIDER',
                 'GROQ_API_KEY','GROQ_MODEL','RETRIEVAL_TOP_K','RETRIEVAL_CANDIDATES']:
        monkeypatch.delenv(name, raising=False)
    values = {'GOOGLE_API_KEY':'embedding-key','PINECONE_API_KEY':'p','PINECONE_INDEX_NAME':'index',
              'LLM_PROVIDER':'groq','GROQ_API_KEY':'chat-key','GROQ_MODEL':'llama-3.3-70b-versatile'}
    result = Settings.load(values)
    assert result.google_api_key == 'embedding-key'
    assert result.groq_api_key == 'chat-key'
    assert result.llm_provider == 'groq'
    assert result.chat_model == 'llama-3.3-70b-versatile'
    del values['GROQ_API_KEY']
    with pytest.raises(DocuSenseError, match='GROQ_API_KEY'):
        Settings.load(values)
