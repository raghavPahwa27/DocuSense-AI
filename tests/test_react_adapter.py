import pytest
from langchain_core.messages import AIMessage
from docusense.react_adapter import to_react_text
from docusense.config import DocuSenseError


def test_native_call_becomes_react_action():
    message = AIMessage(content='', tool_calls=[{'id':'1','name':'calculator','args':{'__arg1':'17 * 23'}}])
    assert to_react_text(message) == 'Action: calculator\nAction Input: 17 * 23'
    assert to_react_text(AIMessage(content='391')) == 'Final Answer: 391'
    assert to_react_text(AIMessage(content='Final Answer: 391')) == 'Final Answer: 391'


def test_invalid_calls_rejected():
    with pytest.raises(DocuSenseError):
        to_react_text(AIMessage(content='',tool_calls=[{'id':'1','name':'shell','args':{'command':'x'}}]))
    with pytest.raises(DocuSenseError):
        to_react_text(AIMessage(content=''))


def test_native_call_runs_through_existing_react_executor(monkeypatch, setup_store):
    from dataclasses import replace
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.prompts import PromptTemplate
    from langchain.memory import ConversationBufferWindowMemory
    from docusense.agent import build_agent
    class NativeFake(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            assert {t.name for t in tools} == {'calculator','document_retrieval','web_search'}
            assert kwargs['parallel_tool_calls'] is False
            return self
    llm = NativeFake(responses=[AIMessage(content='',tool_calls=[{'id':'1','name':'calculator','args':{'__arg1':'17 * 23'}}]), AIMessage(content='391')])
    monkeypatch.setattr('docusense.agent.build_chat_model',lambda settings: llm)
    monkeypatch.setattr('docusense.agent.hub.pull',lambda name: PromptTemplate.from_template('{tools}\n{tool_names}\n{chat_history}\n{input}\n{agent_scratchpad}'))
    factory,_ = setup_store
    store = factory()
    settings = replace(store.settings,llm_provider='groq',chat_model='openai/gpt-oss-20b')
    memory = ConversationBufferWindowMemory(memory_key='chat_history',k=5,return_messages=True,output_key='output')
    agent,_ = build_agent(memory,store,settings)
    result = agent.invoke({'input':'17*23?'})
    assert result['output'] == '391'
    assert result['intermediate_steps'][0][0].tool == 'calculator'
    assert len(memory.chat_memory.messages) == 2
