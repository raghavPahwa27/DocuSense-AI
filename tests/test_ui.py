from pathlib import Path
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
from docusense.config import Settings
from docusense.agent import WebSearchTool

APP = str(Path(__file__).resolve().parent.parent / 'streamlit_app.py')


def test_ui_empty_state_and_session_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, 'load', classmethod(lambda cls, secrets=None: Settings('g','p','test',tmp_path)))
    a = AppTest.from_file(APP).run()
    b = AppTest.from_file(APP).run()
    assert not a.exception
    assert not b.exception
    assert not a.error
    assert not b.error
    assert a.session_state['owner_id'] != b.session_state['owner_id']
    assert any('Upload PDFs to get started' in el.value for el in a.markdown)
    owner = a.session_state['owner_id']
    a.run()
    assert a.session_state['owner_id'] == owner


def test_ui_chat_citations_and_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, 'load', classmethod(lambda cls, secrets=None: Settings('g','p','test',tmp_path)))
    app = AppTest.from_file(APP).run()
    searcher = WebSearchTool()
    class Agent:
        def invoke(self, values):
            searcher.document_evidence = [{'source':'report.pdf','page':0,'chunk_id':'doc:000000'}]
            return {'output':'Revenue was 4500000 [report.pdf | page 1].', 'intermediate_steps':[]}
    app.session_state['agent_executor'] = Agent()
    app.session_state['web_searcher'] = searcher
    app.session_state['doc_names'] = {'doc':'report.pdf'}
    app.run()
    app.text_input(key='question_input').set_value('Revenue?')
    next(b for b in app.button if b.label == 'Ask ✦').click().run()
    assert not app.exception
    assert len(app.session_state['chat_history']) == 1
    assert any('Document Sources' in e.label for e in app.expander)
    app.button(key='clear_btn').click().run()
    assert app.session_state['chat_history'] == []
    assert not app.exception
