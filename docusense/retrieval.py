"""Two-stage retrieval using vector query then separate hosted Inference rerank."""
import math
from langchain_core.documents import Document
from .store import as_dict
from .config import DocuSenseError


class RAGRetriever:
    def __init__(self, store, *, enabled=False, model='bge-reranker-v2-m3', candidates=20, top_k=4):
        if not 1 <= top_k <= candidates <= 100:
            raise DocuSenseError('Require 1 <= top_k <= candidates <= 100.')
        self.store, self.enabled, self.model = store, enabled, model
        self.candidates, self.top_k = candidates, top_k
        self.last_status = 'not_run'
        self._model_available = False

    def retrieve(self, query):
        # Retrieval errors propagate; never turn a provider failure into a fake success.
        docs = self.store.retrieve(query, top_k=self.candidates if self.enabled else self.top_k)
        return self.rank(query, docs)

    def rank(self, query, candidates):
        self.last_status = 'vector'
        fallback = candidates[:self.top_k]
        if not candidates:
            self.last_status = 'empty'
            return []
        if not self.enabled:
            return fallback
        try:
            if not self._model_available:
                available = self.store.client.inference.list_models(type='rerank')
                names = {getattr(m, 'model', None) or as_dict(m).get('model') for m in available}
                if self.model not in names:
                    self.last_status = 'fallback:model_unavailable'
                    return fallback
                self._model_available = True
            response = self.store.client.inference.rerank(
                model=self.model, query=query,
                documents=[{'id':d.metadata['chunk_id'], 'text':d.page_content} for d in candidates],
                rank_fields=['text'], top_n=min(self.top_k, len(candidates)), return_documents=False)
            data = as_dict(response)['data']
            if len(data) != min(self.top_k, len(candidates)):
                raise ValueError('Incomplete ranking')
            ranked, seen = [], set()
            for raw in data:
                row = as_dict(raw)
                i, score = row['index'], row['score']
                if type(i) is not int or i < 0 or i >= len(candidates) or i in seen or not math.isfinite(score):
                    raise ValueError('Invalid ranking')
                seen.add(i)
                original = candidates[i]
                # Reuse original context and metadata, never trust returned document payloads.
                ranked.append(Document(page_content=original.page_content,
                                       metadata=dict(original.metadata, rerank_score=score)))
            self.last_status = 'reranked'
            return ranked
        except Exception:
            self.last_status = 'fallback:api_or_response_error'
            return fallback


def configured_retriever(store, settings):
    return RAGRetriever(store, enabled=settings.rerank_enabled, model=settings.rerank_model,
                        candidates=settings.candidate_k, top_k=settings.top_k)
