#!/usr/bin/env python3
"""Offline report from optional Gemini 3.8 Flash experiment JSON outputs.

Does not send requests, need credentials, or reveal transcript/quote content.
Requires only the standard library and (optionally) CQA's benchmark module.
"""
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def percentile(values, percent):
    """Nearest-rank percentile, like cqa.benchmark."""
    if not values:
        return None
    xs = sorted(values)
    return xs[max(0, min(len(xs) - 1, math.ceil(len(xs) * percent) - 1))]


def valid_metric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def summarize(path, input_price=None, output_price=None):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Ожидается JSON-массив результатов.')
    ids = [r.get('conversation_id') for r in rows]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError('ID диалогов отсутствуют или повторяются.')
    successes = [r for r in rows if not r.get('error') and isinstance(r.get('assessment'), dict)
                 and isinstance(r['assessment'].get('criteria'), dict) and bool(r['assessment']['criteria'])
                 and isinstance(r.get('score'), dict) and 'total' in r['score']]
    success_ids = {r['conversation_id'] for r in successes}
    failure = [r for r in rows if r['conversation_id'] not in success_ids]
    inputs = 0
    outputs = 0
    times = []
    stages = Counter()
    attempts = 0
    count_with_tokens = 0
    changed = 0
    for row in successes:
        usage = row.get('usage')
        if not isinstance(usage, dict):
            continue
        if valid_metric(usage.get('seconds')):
            times.append(float(usage['seconds']))
        a, b = usage.get('input_tokens'), usage.get('output_tokens')
        if type(a) is int and type(b) is int and a >= 0 and b >= 0:
            count_with_tokens += 1
            inputs += a
            outputs += b
        calls = usage.get('per_call')
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    stage = call.get('stage')
                    if isinstance(stage, str) and stage:
                        stages[stage] += 1
                    n = call.get('http_attempts', 1)
                    if type(n) is int and n > 0:
                        attempts += n
        if row.get('review_changes'):
            changed += 1
    if input_price is not None and (not valid_metric(input_price) or not valid_metric(output_price)):
        raise ValueError('Обе цены должны быть конечными неотрицательными числами.')
    cost = (inputs * input_price + outputs * output_price) / 1_000_000 if input_price is not None else None
    return {
        'source_filename': Path(path).name,
        'selected_records': len(rows),
        'successful': len(successes),
        'failed_or_incomplete': len(failure),
        'success_rate': len(successes)/len(rows) if rows else None,
        'error_kinds': dict(Counter('provider_or_validation_error' if r.get('error') else 'incomplete_result' for r in failure)),
        'review_changed_records': changed,
        'calls_by_stage': dict(stages),
        'total_http_attempts_if_reported': attempts,
        'token_usage': {'measured_records': count_with_tokens, 'input_tokens': inputs, 'output_tokens': outputs,
                        'total_tokens': inputs + outputs},
        'provider_latency_seconds': {
            'measured_records': len(times),
            'total_sum': round(sum(times), 2),
            'mean': round(statistics.mean(times), 2) if times else None,
            'median_p50': round(statistics.median(times), 2) if times else None,
            'p95_nearest_rank': round(percentile(times, 0.95), 2) if times else None,
            'max': round(max(times), 2) if times else None,
        },
        'estimated_cost_usd': round(cost, 8) if cost is not None else None,
        'cost_assumptions': {'input_usd_per_1m': input_price, 'output_usd_per_1m': output_price,
                             'note': 'Price covers recorded successful results only; failed calls may be billed. Provider billing can differ due to retries, cached/reasoning tokens, or pricing tiers.'},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Сводная статистика CQA Gemini 3.8: только локальные JSON, без сети')
    parser.add_argument('files', nargs='+', type=Path, help='Один или несколько файлов data/gemini38_*.json')
    parser.add_argument('--input-price', type=float, help='Тариф на 1M входных токенов, USD')
    parser.add_argument('--output-price', type=float, help='Тариф на 1M выходных токенов, USD')
    parser.add_argument('--output', type=Path, help='Необязательный приватный JSON-отчёт')
    args = parser.parse_args(argv)
    if (args.input_price is None) != (args.output_price is None):
        parser.error('Укажите одновременно --input-price и --output-price либо не указывайте цены.')
    try:
        report = {'experiments': [summarize(f, args.input_price, args.output_price) for f in args.files],
                  'ground_truth_accuracy': None,
                  'note': 'Скорость/токены не измеряют близость к Ground Truth. Для точности запускайте manage.py benchmark с независимой разметкой.'}
        if args.output:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from manage import write_private_json
            write_private_json(args.output, report)
            print(f'Отчёт сохранён: {args.output}')
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
