"""Shared chat-model factory; embedding configuration remains independent."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def build_chat_model(settings):
    if settings.llm_provider == 'groq':
        return ChatGroq(model=settings.chat_model, api_key=settings.groq_api_key,
                        temperature=0, timeout=30, max_retries=1)
    return ChatGoogleGenerativeAI(model=settings.chat_model,
                                 google_api_key=settings.google_api_key, temperature=0)
