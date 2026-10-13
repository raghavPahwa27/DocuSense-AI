from docusense.calculator import calculate
from docusense.agent import WebSearchTool
from docusense.agent import (
    DynamicAgentExecutor,
    _AGENT_GROUNDING_INSTRUCTIONS,
    _simple_date_answer,
    _now_context,
)


def test_calculator_preserved():
    assert calculate('0.15 * 4500000').strip() == '675000.0'
    assert calculate('print((10 + 5) / 3)').strip() == '5.0'


def test_calculator_cannot_bypass_document_isolation():
    assert 'accepts numeric' in calculate("__import__('os').environ")
    assert 'accepts numeric' in calculate("open('/tmp/test')")
    assert 'accepts numeric' in calculate('2 ** 1000000000')


def test_web_evidence_and_reset(monkeypatch):
    class DDGS:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, *args, **kwargs):
            return [{'href':'https://example.com', 'title':'Example', 'body':'Evidence'}]
    monkeypatch.setattr('docusense.agent._DDGS', DDGS)
    tool = WebSearchTool()
    assert 'Evidence' in tool.run('query')
    tool.run('query')
    assert len(tool.evidence) == 1
    tool.reset()
    assert tool.evidence == []


def test_date_answers_are_deterministic_and_do_not_use_web(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    class Memory:
        def __init__(self): self.saved = []
        def save_context(self, inputs, outputs): self.saved.append((inputs, outputs))
    class Executor:
        def __init__(self): self.memory = Memory(); self.called = False
        def invoke(self, inputs, **kwargs): self.called = True; return {'output': 'wrong'}
    executor = Executor()
    result = DynamicAgentExecutor(executor).invoke({'input': "What is today's day?"})
    assert result['output'].startswith('Today is ')
    expected = datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%A, %B %-d, %Y')
    assert expected in result['output']
    assert not executor.called
    assert executor.memory.saved
    assert 'IST' in _now_context()
    assert _simple_date_answer("What is today's date?").startswith('Today is ')


def test_empty_web_results_are_explicitly_non_evidence(monkeypatch):
    class DDGS:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, *args, **kwargs): return []
    monkeypatch.setattr('docusense.agent._DDGS', DDGS)
    result = WebSearchTool().run('query')
    assert 'WEB_SEARCH_NO_EVIDENCE' in result
    assert 'Do not infer facts' in result


def test_web_exceptions_are_explicitly_non_evidence(monkeypatch):
    class DDGS:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, *args, **kwargs): raise RuntimeError('provider detail')
    monkeypatch.setattr('docusense.agent._DDGS', DDGS)
    result = WebSearchTool().run('query')
    assert 'WEB_SEARCH_UNAVAILABLE' in result
    assert 'provider detail' not in result


class _RecordingExecutor:
    def __init__(self, output):
        self.memory = type('Memory', (), {'save_context': lambda *args, **kwargs: None})()
        self.output = output
        self.inputs = []

    def invoke(self, inputs, **kwargs):
        self.inputs.append(inputs)
        return {'input': inputs['input'], 'output': self.output, 'intermediate_steps': []}


def test_multipart_document_and_date_question_keeps_both_parts():
    executor = _RecordingExecutor(
        '1. Exam city preference: use the application options shown in the document.\n'
        '2. Today\'s date: Monday, October 12, 2026.'
    )
    question = "What is exam city preference in the application and what's today's date?"
    result = DynamicAgentExecutor(executor).invoke({'input': question})
    assert 'Exam city preference' in result['output']
    assert "Today's date" in result['output']
    assert executor.inputs[0]['current_datetime'].endswith(' IST')


def test_two_unrelated_general_questions_are_preserved_as_separate_parts():
    executor = _RecordingExecutor('1. The sky appears blue due to scattered light.\n2. 2 + 2 = 4.')
    result = DynamicAgentExecutor(executor).invoke(
        {'input': 'Why is the sky blue, and what is 2 + 2?'}
    )
    assert result['output'].startswith('1.')
    assert '2 + 2 = 4' in result['output']
    assert len(executor.inputs) == 1


def test_document_question_combined_with_date_is_not_reduced_to_date_only():
    executor = _RecordingExecutor(
        '1. The application requires the information stated in the retrieved document.\n'
        '2. Today is Monday, October 12, 2026.'
    )
    result = DynamicAgentExecutor(executor).invoke(
        {'input': 'What information is required in the application, and what is today\'s date?'}
    )
    assert 'retrieved document' in result['output']
    assert 'Monday, October 12, 2026' in result['output']
    assert executor.inputs[0]['input'].startswith('What information')


def test_multipart_answer_can_explain_one_unverified_part_while_answering_another():
    executor = _RecordingExecutor(
        '1. 15% of 200 is 30.\n'
        '2. I cannot verify the requested application deadline because it is not in the retrieved context.'
    )
    result = DynamicAgentExecutor(executor).invoke(
        {'input': 'What is 15% of 200, and what is the application deadline?'}
    )
    assert '15% of 200 is 30' in result['output']
    assert 'cannot verify' in result['output']


def test_prompt_requires_all_multipart_parts_and_explicit_uncertainty():
    prompt = _AGENT_GROUNDING_INSTRUCTIONS.lower()
    assert 'every distinct question or sub-question' in prompt
    assert 'numbered points or' in prompt
    assert 'answer the remaining parts' in prompt
    assert 'missing' in prompt


def test_real_react_executor_tools_memory_and_citations(monkeypatch, setup_store):
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain.memory import ConversationBufferWindowMemory
    from docusense.agent import build_agent
    from conftest import chunk
    factory, _ = setup_store
    store = factory()
    store.ingest_chunks('doc-a', [chunk()])
    llm = FakeListLLM(responses=[
        'Thought: Retrieve.\nAction: document_retrieval\nAction Input: revenue',
        'Thought: Calculate.\nAction: calculator\nAction Input: 0.15 * 4500000',
        'Thought: Finished.\nFinal Answer: 675000.0, based on revenue [report.pdf | page 1].',
        'Thought: Follow up.\nFinal Answer: The prior result was 675000.0.'
    ])
    monkeypatch.setattr('docusense.agent.build_chat_model', lambda settings: llm)
    prompt = PromptTemplate.from_template('Tools: {tools}\nNames: {tool_names}\nHistory: {chat_history}\nQuestion: {input}\n{agent_scratchpad}')
    monkeypatch.setattr('docusense.agent.hub.pull', lambda name: prompt)
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True, output_key='output')
    executor, evidence = build_agent(memory, store, store.settings)
    result = executor.invoke({'input':'What is 15% of revenue?'})
    assert '[report.pdf | page 1]' in result['output']
    assert [a.tool for a, _ in result['intermediate_steps']] == ['document_retrieval','calculator']
    assert '[report.pdf | page 1 | chunk doc-a:000000]' in result['intermediate_steps'][0][1]
    assert evidence.document_evidence[0]['document_id'] == 'doc-a'
    evidence.reset()
    executor.invoke({'input':'What was that result?'})
    assert len(memory.chat_memory.messages) == 4
    assert '675000.0' in str(memory.load_memory_variables({})['chat_history'])
    for i in range(6):
        memory.save_context({'input':f'Question {i}'},{'output':f'Answer {i}'})
    assert len(memory.load_memory_variables({})['chat_history']) == 10
    memory.clear()
    assert memory.load_memory_variables({})['chat_history'] == []
    assert evidence.document_evidence == []
