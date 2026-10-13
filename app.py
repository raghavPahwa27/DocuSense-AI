"""CLI using the same Pinecone retrieval, tools, ReAct agent and memory as the UI."""
import argparse
from pathlib import Path
from langchain.memory import ConversationBufferWindowMemory
from docusense.config import Settings, DocuSenseError
from docusense.store import connect_store
from docusense.agent import build_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-id', required=True, help='Private stable owner ID for this local CLI operator')
    parser.add_argument('--pdf', type=Path, action='append', default=[])
    args = parser.parse_args()
    try:
        settings = Settings.load()
        store = connect_store(settings, args.owner_id)
        for path in args.pdf:
            result = store.ingest_pdf(path)
            print(f'Indexed {result.chunk_count} chunks; duplicate={result.duplicate}')
        memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True, output_key='output')
        agent, searcher = build_agent(memory, store, settings)
        while True:
            query = input('\nYou: ').strip()
            if query.lower() in {'exit', 'quit'}:
                break
            if query:
                searcher.reset()
                try:
                    result = agent.invoke({'input': query})
                    print(result['output'])
                except Exception:
                    print('Agent request failed. Check chat-provider availability and retry.')
    except DocuSenseError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
