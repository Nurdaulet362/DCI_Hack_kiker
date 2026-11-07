#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from cqa.benchmark import compare
from cqa.datasets import fetch_dataset
from cqa.evaluator import evaluate
from cqa.provider import Provider, ProviderError
from cqa.settings import ROOT
from cqa.storage import Store


def main():
    parser = argparse.ArgumentParser(description='CQA: import, evaluate, benchmark')
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch'); fetch.add_argument('--group', default='DEV')
    imp = sub.add_parser('import'); imp.add_argument('path', type=Path)
    run = sub.add_parser('evaluate'); run.add_argument('--id', action='append')
    run.add_argument('--limit', type=int, default=1); run.add_argument('--all', action='store_true')
    run.add_argument('--checklist'); run.add_argument('--output', type=Path, default=ROOT/'data/predictions.json')
    bench = sub.add_parser('benchmark'); bench.add_argument('predictions', type=Path); bench.add_argument('ground_truth', type=Path)
    args = parser.parse_args()
    if args.command == 'benchmark':
        print(json.dumps(compare(json.loads(args.predictions.read_text()), json.loads(args.ground_truth.read_text())), ensure_ascii=False, indent=2))
        return
    store = Store()
    if args.command in ['fetch', 'import']:
        payload = fetch_dataset(args.group) if args.command == 'fetch' else json.loads(args.path.read_text())
        count = store.import_records(payload)
        if args.command == 'fetch' and args.group == 'DEV':
            (ROOT/'data/dev.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f'Imported {count} conversations. Raw transcripts stay in ignored data/.')
    elif args.command == 'evaluate':
        provider = Provider()
        checklist = store.checklist(args.checklist)
        ids = args.id or [c['conversation_id'] for c in store.conversations()]
        if not args.id and not args.all: ids = ids[:max(1, args.limit)]
        results = []
        for cid in ids:
            try:
                conv = store.conversation(cid)
                result = evaluate(conv, checklist, provider, review=provider.cfg['review'])
                saved = store.save_evaluation(conv, checklist, result)
                results.append(saved)
                print(f'{cid}: {saved["score"]["total"]}')
            except (ValueError, ProviderError) as e:
                results.append({'conversation_id': cid, 'error': str(e)})
                print(f'{cid}: FAILED: {e}')
            # Save after every record so a long batch survives interruptions.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try: main()
    except (ValueError, ProviderError, OSError) as e: raise SystemExit(str(e))
