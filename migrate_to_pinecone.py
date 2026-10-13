"""Re-embed original PDFs; never read, mutate, or deserialize the FAISS store."""
import argparse
import json
from pathlib import Path
from docusense.config import Settings, DocuSenseError
from docusense.store import connect_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-id', required=True, help='Explicit owner for every selected original PDF')
    parser.add_argument('--pdf', type=Path, action='append', required=True)
    parser.add_argument('--source', help='Original citation filename (only with one --pdf)')
    parser.add_argument('--reindex', action='store_true', help='Re-embed and verify before publishing the replacement')
    parser.add_argument('--report', type=Path, default=Path('.docusense/migration-report.json'))
    args = parser.parse_args()
    if args.source and len(args.pdf) != 1:
        parser.error('--source requires exactly one --pdf')
    rows = []
    try:
        store = connect_store(Settings.load(), args.owner_id)
        for path in args.pdf:
            try:
                result = store.ingest_pdf(path, args.source, reindex=args.reindex)
                rows.append(dict(document_id=result.document_id, chunk_count=result.chunk_count,
                                 duplicate=result.duplicate, verified=True))
            except DocuSenseError as exc:
                rows.append(dict(verified=False, error=str(exc)))
        args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.report.write_text(json.dumps({'owner_id': args.owner_id, 'dimension': store.dimension,
            'faiss_untouched': True, 'documents': rows}, indent=2))
        args.report.chmod(0o600)
        print(f'Verified {sum(r["verified"] for r in rows)}/{len(rows)} documents. Report: {args.report}')
        return 0 if all(r['verified'] for r in rows) else 1
    except DocuSenseError as exc:
        print(str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
