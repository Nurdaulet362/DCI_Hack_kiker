"""Frozen ten-dialogue FBroker run. No inference/network calls on import.

Uniform model/protocol, immutable input/checklist hashes and checkpointed errors.
SWAP is intentionally not a supported input of this runner.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import run as base
import retry_consistency as consistency
from cqa import evaluator
from cqa.domain import digest, validate_checklist

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'fbroker-control-v1-uniform-proof-identity'
MODEL_DIGEST = base.MODEL_DIGEST
REVIEW = 'auto'


def runtime():
    result = consistency.runtime()
    result.update(evidence_protocol=PROTOCOL,
                  control_runner_hash=digest(Path(__file__).read_text()),
                  routing_source_hash=digest((ROOT/'cqa/routing.py').read_text()),
                  review_policy=REVIEW, uniform_all_dialogues=True,
                  trace_file='consistency-trace.jsonl')
    return result


def experiment_config(provider, review=REVIEW):
    if review != REVIEW: raise ValueError('Control review policy is frozen to auto')
    # base.BASE_EXPERIMENT captured the original function before wrapper installation.
    result = base.experiment_config(provider, review)
    result.update(prompt_version=evaluator.PROMPT_VERSION + '+' + PROTOCOL,
                  output_protocol=PROTOCOL, runtime=runtime())
    return result


def verify_manifest():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise SystemExit('Frozen payload modified: ' + name)


def inputs():
    conversations = json.loads((ROOT/'conversations.json').read_text())
    checklist = json.loads((ROOT/'checklist.json').read_text())
    validate_checklist(checklist)
    ids = [c['conversation_id'] for c in conversations]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise SystemExit('Expected exactly 10 unique control dialogues')
    return conversations, checklist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    verify_manifest()
    conversations, checklist = inputs()
    if args.validate_only:
        schema = evaluator.response_schema(checklist)
        for conv in conversations:
            consistency.consistent_schema(schema, conv, checklist)
        print('OFFLINE_VALIDATED 10 inputs, frozen checklist and decoder schemas; no API calls')
        return
    deadline = float(os.environ['RUN_DEADLINE_UNIX'])
    if not time.time() < deadline: raise SystemExit('Deadline must be in the future')
    provider = consistency.ConsistencyProvider(checklist, deadline)
    with urllib.request.urlopen(provider.cfg['base_url']+'/api/tags', timeout=min(20, deadline-time.time())) as response:
        models = json.load(response).get('models', [])
    model = next((m for m in models if m.get('name') == provider.cfg['model']), None)
    if not model or model.get('digest') != MODEL_DIGEST:
        raise SystemExit('Expected pinned Qwen model digest unavailable')
    experiment = experiment_config(provider)
    base.atomic_json(ROOT/'experiment.json', experiment)
    output = ROOT/'predictions.json'
    if output.exists() and not args.resume:
        raise SystemExit('Predictions already exist; --resume required to retain records')
    records = json.loads(output.read_text()) if output.exists() else []
    by_id = {c['conversation_id']: c for c in conversations}
    if len({r['conversation_id'] for r in records}) != len(records) or any(r['conversation_id'] not in by_id for r in records):
        raise SystemExit('Foreign or duplicate prediction IDs')
    results = {r['conversation_id']: r for r in records}
    for conv in conversations:
        cid = conv['conversation_id']
        if cid not in results:
            results[cid] = {'conversation_id': cid, 'error': 'Control model evaluation not yet attempted',
                           'transcript_hash': digest(conv), 'checklist_hash': digest(checklist)}
    def checkpoint():
        base.atomic_json(output, [results[c['conversation_id']] for c in conversations])
    checkpoint()
    previous_config = evaluator.experiment_config
    evaluator.experiment_config = experiment_config
    try:
        for conv in conversations:
            cid = conv['conversation_id']
            if base.reusable(results[cid], conv, checklist, experiment):
                print('REUSED', cid, flush=True); continue
            if deadline-time.time() < 90:
                print('DEADLINE: completed records preserved; unattempted records remain errors', flush=True); break
            provider.conversation = conv; provider.calls = []
            record = {'conversation_id': cid, 'language': conv['language'], 'channel': conv['channel'],
                      'transcript_hash': digest(conv), 'checklist_hash': digest(checklist),
                      'experiment_config': experiment, 'experiment_hash': digest(experiment), 'runtime': runtime()}
            print('START', cid, flush=True)
            try:
                result = evaluator.evaluate(conv, checklist, provider=provider, review=REVIEW,
                                           progress=lambda stage: print(cid, stage, flush=True))
                base.validate(result['assessment'], checklist, conv)
                base.validate(result['draft'], checklist, conv)
                result['prompt_version'] = experiment['prompt_version']
                result['base_prompt_version'] = evaluator.PROMPT_VERSION
                record.update(result)
                print('DONE', cid, result['score']['total'], 'calls', len(provider.calls), flush=True)
            except Exception as err:
                record['error'] = type(err).__name__+': '+str(err)
                print('FAILED', cid, record['error'], flush=True)
            record['all_calls_usage'] = {
                'calls': len(provider.calls),
                'input_tokens': sum(c['usage']['input_tokens'] for c in provider.calls),
                'output_tokens': sum(c['usage']['output_tokens'] for c in provider.calls),
                'seconds': round(sum(c['usage']['seconds'] for c in provider.calls),2), 'cost': None}
            results[cid] = record; checkpoint()
    finally:
        evaluator.experiment_config = previous_config
    successes = sum(not r.get('error') for r in results.values())
    print('FINISHED', successes, 'successful out of 10', flush=True)
    if successes != 10: raise SystemExit(2)


if __name__ == '__main__': main()
