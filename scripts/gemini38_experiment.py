#!/usr/bin/env python3
"""Opt-in, isolated Gemini 3.8 Flash experiment for CQA.

No changes to server.py, SQLite evaluations, checklists, or existing .env.
The program defaults to a no-network preview and stores outcomes in ignored data/.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Permit 'python scripts/gemini38_experiment.py' from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.evaluator import evaluate, experiment_config
from cqa.provider import Provider, ProviderError
from cqa.settings import ROOT  # loads existing .env, without rewriting it
from cqa.storage import Store
from manage import reusable_result, write_private_json

PROFILE = {
    'AI_PROVIDER': 'openai_compatible',
    'AI_BASE_URL': 'https://generativelanguage.googleapis.com/v1beta/openai',
    'AI_MODEL': 'gemini-3.8-flash',
    'AI_REASONING_EFFORT': 'low',
}


def activate_profile(review):
    """Changes environment for THIS process only; never touches .env or Git files."""
    os.environ.update(PROFILE)
    os.environ['AI_REVIEW'] = 'true' if review else 'false'
    # Unknown/stale model tariffs must not produce an apparently exact USD price.
    os.environ.pop('AI_INPUT_PRICE', None)
    os.environ.pop('AI_OUTPUT_PRICE', None)


def select_ids(summaries, explicit_ids, limit, all_records):
    available = [s['conversation_id'] for s in summaries]
    if explicit_ids:
        if len(set(explicit_ids)) != len(explicit_ids):
            raise ValueError('Повторяющиеся --id.')
        missing = [cid for cid in explicit_ids if cid not in available]
        if missing:
            raise ValueError(f'Диалоги не найдены: {len(missing)}.')
        return explicit_ids
    if limit < 1:
        raise ValueError('--limit должен быть не меньше 1.')
    return available if all_records else available[:limit]


def read_checkpoint(path):
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list) or any(not isinstance(row, dict)
                                            or not isinstance(row.get('conversation_id'), str)
                                            for row in records):
        raise ValueError('Файл результатов повреждён; выберите новый --output.')
    ids = [row['conversation_id'] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Дублирующиеся conversation_id в файле результатов.')
    return {row['conversation_id']: row for row in records}


def evaluate_one(conversation, checklist, review):
    """One provider per thread; reuse the application's validation and scoring."""
    cid = conversation['conversation_id']
    try:
        provider = Provider()
        outcome = evaluate(conversation, checklist, provider=provider, review=review)
        return {'conversation_id': cid, **outcome}
    except (ProviderError, ValueError) as exc:
        return {'conversation_id': cid, 'error': str(exc)}
    except Exception:
        # Avoid leaking response bodies, keys or transcript text in exception messages.
        return {'conversation_id': cid, 'error': 'Непредвиденная ошибка; проверьте локальный журнал.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Gemini 3.8 Flash: безопасное измерение скорости CQA без изменения приложения')
    parser.add_argument('--id', action='append', help='Идентификатор диалога; можно указать несколько раз')
    parser.add_argument('--limit', type=int, default=3, help='Первые N диалогов; по умолчанию 3')
    parser.add_argument('--all', action='store_true', help='Явно выбрать весь импортированный набор')
    parser.add_argument('--checklist', help='ID версии чек-листа, если нужна не последняя')
    parser.add_argument('--review', choices=['on', 'off'], default='on', help='Два прохода (on) или один (off)')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1, help='Параллельных диалогов, максимум 2')
    parser.add_argument('--output', type=Path, help='JSON результатов; только новый файл либо --resume')
    parser.add_argument('--resume', action='store_true', help='Повторно использовать только валидные результаты того же эксперимента')
    parser.add_argument('--execute', action='store_true', help='Реально отправлять запросы API; без флага только предпросмотр')
    parser.add_argument('--allow-cloud-data', action='store_true',
                        help='Подтверждает разрешение отправлять выбранные транскрипты внешнему AI')
    args = parser.parse_args(argv)
    review = args.review == 'on'
    activate_profile(review)
    if args.execute and not args.allow_cloud_data:
        parser.error('Для отправки транскриптов в Google API требуется --allow-cloud-data.')
    output = args.output or (ROOT / 'data' / f'gemini38_review_{args.review}.json')
    if output.exists() and not args.resume:
        parser.error('Файл --output уже существует; используйте --resume или другое имя.')
    try:
        store = Store()
        checklist = store.checklist(args.checklist)
        ids = select_ids(store.conversations(), args.id, args.limit, args.all)
        if not ids:
            raise ValueError('Нет диалогов; сначала импортируйте DEV-набор.')
        conversations = {cid: store.conversation(cid) for cid in ids}
        # No API request before --execute. Metadata-only configuration is safe.
        if args.execute:
            provider = Provider()
            experiment = experiment_config(provider, review)
        else:
            experiment = None
        results = read_checkpoint(output) if args.resume else {}
        reused = set()
        if args.execute and args.resume:
            for cid in ids:
                if reusable_result(results.get(cid), conversations[cid], checklist, experiment):
                    reused.add(cid)
        pending = [cid for cid in ids if cid not in reused]
        preview = {'model': PROFILE['AI_MODEL'], 'reasoning': 'low', 'review': args.review,
                   'workers': args.workers, 'selected': len(ids), 'reused': len(reused),
                   'to_evaluate': len(pending), 'output': str(output), 'network_calls_enabled': args.execute}
        print(json.dumps(preview, ensure_ascii=False), flush=True)
        if not args.execute:
            print('Предпросмотр: запросы AI не выполнялись, результаты не записаны.')
            return 0
        started = time.monotonic()
        # Concurrent *independent conversations*. Inside a conversation, the reviewer
        # still runs after the initial pass. SQLite is read before worker threads run.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate_one, conversations[cid], checklist, review): cid for cid in pending}
            for future in as_completed(futures):
                cid = futures[future]
                result = future.result()
                results[cid] = result
                # Single writer in main thread; atomic + private like regular CLI.
                write_private_json(output, list(results.values()))
                outcome = 'FAILED' if result.get('error') else 'OK'
                print(f'{cid}: {outcome}', flush=True)
        print(json.dumps({'completed_now': sum(not results[cid].get('error') for cid in pending),
                          'failed_now': sum(bool(results[cid].get('error')) for cid in pending),
                          'reused': len(reused), 'wall_seconds': round(time.monotonic()-started, 2),
                          'output': str(output)}, ensure_ascii=False), flush=True)
        return 0 if all(not results[cid].get('error') for cid in ids) else 1
    except (ProviderError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
