#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from pathlib import Path

from cqa.benchmark import compare
from cqa.datasets import fetch_dataset
from cqa.domain import choices, digest, validate_assessment, calculate
from cqa.evaluator import evaluate, experiment_config
from cqa.quality import validate_proofs
from cqa.provider import Provider, ProviderError
from cqa.settings import ROOT
from cqa.storage import Store


def write_private_json(path, value):
    """Atomic checkpoints, private permissions, no half-written prediction files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, delete=False) as f:
            temporary = Path(f.name)
            if os.name == 'posix': os.fchmod(f.fileno(), 0o600)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary and temporary.exists(): temporary.unlink()


def reusable_result(record, conversation, checklist, experiment):
    """A checkpoint is untrusted: hashes alone do not establish a valid result."""
    if not (isinstance(record, dict) and not record.get('error')
            and record.get('transcript_hash') == digest(conversation)
            and record.get('checklist_hash') == digest(checklist)
            and record.get('experiment_hash') == digest(experiment)):
        return False
    try:
        # A stored export may carry snapshots. They must match the current inputs,
        # rather than attaching old evidence to a copied current hash.
        for key, expected in [('conversation', conversation), ('checklist', checklist),
                              ('experiment_config', experiment)]:
            if key in record and digest(record[key]) != digest(expected):
                return False
        assessment = validate_assessment(record.get('assessment'), checklist, conversation)
        validate_proofs(assessment, checklist, conversation)
        recalculated = calculate(checklist, assessment)
        return isinstance(record.get('score'), dict) and digest(record['score']) == digest(recalculated)
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def label_template(conversations, checklist):
    """Unlabelled annotation pack. Never expose previous AI answers to annotators."""
    return [{'conversation_id': c['conversation_id'], 'language': c['language'],
             'channel': c['channel'], 'transcript': c['transcript'],
             'checklist': checklist, 'checklist_hash': digest(checklist),
             'transcript_hash': digest(c),
             'answer_options': {rule['id']: choices(rule) for rule in checklist['criteria']},
             'criteria': {rule['id']: None for rule in checklist['criteria']},
             'total': None, 'annotator': '', 'notes': ''} for c in conversations]


def main():
    parser = argparse.ArgumentParser(description='CQA: import, evaluate, label-template, benchmark')
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch'); fetch.add_argument('--group', default='DEV')
    imp = sub.add_parser('import'); imp.add_argument('path', type=Path)
    run = sub.add_parser('evaluate'); run.add_argument('--id', action='append')
    run.add_argument('--limit', type=int, default=1); run.add_argument('--all', action='store_true')
    run.add_argument('--review-mode', choices=['auto', 'full', 'single'], default=None,
                     help='auto: второй проход только для рискованных случаев; full: всегда; single: один проход.')
    run.add_argument('--resume', action='store_true', help='Skip successful identical transcript/checklist/experiment; retry errors.')
    run.add_argument('--checklist'); run.add_argument('--output', type=Path, default=ROOT/'data/predictions.json')
    label = sub.add_parser('label-template', help='Export blank independent human annotation pack; no AI calls or AI answers.')
    label.add_argument('--id', action='append'); label.add_argument('--limit', type=int, default=20)
    label.add_argument('--checklist'); label.add_argument('--output', type=Path, default=ROOT/'data/ground_truth_template.json')
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
            write_private_json(ROOT/'data/dev.json', payload)
        print(f'Imported {count} conversations. Raw transcripts stay in ignored data/.')
    elif args.command == 'label-template':
        if args.output.exists(): raise ValueError('Файл разметки уже существует. Выберите другой --output, чтобы не потерять работу эксперта.')
        checklist = store.checklist(args.checklist)
        # Round-robin languages provides a useful initial mix, not a representative statistical sample.
        if args.id:
            ids = args.id
        else:
            groups = {language: [c['conversation_id'] for c in store.conversations() if c['language'] == language]
                      for language in ['RU', 'KZ', 'EN']}
            ids = []
            while any(groups.values()) and len(ids) < max(1, args.limit):
                for group in groups.values():
                    if group and len(ids) < max(1, args.limit): ids.append(group.pop(0))
        records = label_template([store.conversation(cid) for cid in ids], checklist)
        write_private_json(args.output, records)
        print(f'Exported {len(records)} unlabelled records. An independent reviewer must fill criteria and total; do not use AI predictions as Ground Truth.')
    elif args.command == 'evaluate':
        provider = Provider()
        checklist = store.checklist(args.checklist)
        review = {'auto': 'auto', 'full': True, 'single': False}.get(args.review_mode, provider.cfg['review'])
        experiment = experiment_config(provider, review)
        ids = args.id or [c['conversation_id'] for c in store.conversations()]
        if not args.id and not args.all: ids = ids[:max(1, args.limit)]
        if len(set(ids)) != len(ids): raise ValueError('Повторяющиеся ID в запросе.')
        old = json.loads(args.output.read_text()) if args.output.exists() else []
        if not isinstance(old, list) or any(not isinstance(r, dict) or 'conversation_id' not in r for r in old):
            raise ValueError('Существующий --output имеет неверный формат. Выберите новый файл.')
        results = {r['conversation_id']: r for r in old}
        for cid in ids:
            try:
                conv = store.conversation(cid)
                previous = results.get(cid)
                if args.resume:
                    if not reusable_result(previous, conv, checklist, experiment):
                        previous = next((record for h in store.history(cid)
                            if reusable_result(record := store.evaluation(h['id']), conv, checklist, experiment)), None)
                    if previous is not None and reusable_result(previous, conv, checklist, experiment):
                        results[cid] = previous
                        write_private_json(args.output, list(results.values()))
                        print(f'{cid}: reused validated result (no API call)', flush=True)
                        continue
                result = evaluate(conv, checklist, provider, review=review,
                                  progress=lambda message: print(f'{cid}: {message}', flush=True))
                saved = store.save_evaluation(conv, checklist, result)
                results[cid] = saved
                print(f'{cid}: {saved["score"]["total"]}', flush=True)
            except (ValueError, ProviderError) as e:
                # Prior successful results remain in immutable SQLite history.
                results[cid] = {'conversation_id': cid, 'error': str(e)}
                print(f'{cid}: FAILED: {e}', flush=True)
            write_private_json(args.output, list(results.values()))
        failed = sum(bool(results.get(cid, {}).get('error')) for cid in ids)
        if failed: raise ProviderError(f'Не удалось оценить {failed} из {len(ids)} диалогов. Подробности сохранены в --output.')


if __name__ == '__main__':
    try: main()
    except (ValueError, ProviderError, OSError) as e: raise SystemExit(str(e))
