"""Configuration without import-time network calls or secret logging."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping, Optional
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = 'models/gemini-embedding-001'
LLM_MODEL = 'gemini-3.6-flash'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

class DocuSenseError(Exception):
    """A safe, user-facing error; never include provider response bodies."""

@dataclass(frozen=True)
class Settings:
    google_api_key: str = field(repr=False)
    pinecone_api_key: str = field(repr=False)
    index_name: str
    state_dir: Path = ROOT / '.docusense'
    rerank_enabled: bool = False
    rerank_model: str = 'bge-reranker-v2-m3'
    candidate_k: int = 20
    top_k: int = 4
    llm_provider: str = 'gemini'
    chat_model: str = LLM_MODEL
    groq_api_key: str = field(default='', repr=False)

    @classmethod
    def load(cls, secrets: Optional[Mapping] = None):
        load_dotenv(ROOT / '.env')
        def get(name, default=''):
            return os.environ.get(name) or (secrets or {}).get(name, default)
        required = ['GOOGLE_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
        values = [get(key) for key in required]
        missing = [key for key, value in zip(required, values) if not value]
        if missing:
            raise DocuSenseError('Configure ' + ', '.join(missing) + ' in .env or Streamlit secrets.')
        provider = str(get('LLM_PROVIDER', 'gemini')).lower()
        if provider not in ('gemini', 'groq'):
            raise DocuSenseError('LLM_PROVIDER must be gemini or groq.')
        groq_key = get('GROQ_API_KEY')
        model = get('GROQ_MODEL') if provider == 'groq' else LLM_MODEL
        if provider == 'groq' and (not groq_key or not model):
            raise DocuSenseError('Configure GROQ_API_KEY and GROQ_MODEL to use Groq chat. Keep GOOGLE_API_KEY for embeddings.')
        try:
            enabled = str(get('RERANK_ENABLED', 'false')).lower()
            candidates, top_k = int(get('RETRIEVAL_CANDIDATES', 20)), int(get('RETRIEVAL_TOP_K', 4))
            if enabled not in ('true', 'false') or not 1 <= top_k <= candidates <= 100:
                raise ValueError
        except (TypeError, ValueError):
            raise DocuSenseError('Use RERANK_ENABLED=true/false and 1 <= RETRIEVAL_TOP_K <= RETRIEVAL_CANDIDATES <= 100.') from None
        return cls(*values, state_dir=Path(get('DOCUSENSE_STATE_DIR', str(ROOT / '.docusense'))),
                   rerank_enabled=enabled == 'true', rerank_model=get('PINECONE_RERANK_MODEL', 'bge-reranker-v2-m3'),
                   candidate_k=candidates, top_k=top_k, llm_provider=provider, chat_model=model,
                   groq_api_key=groq_key)
