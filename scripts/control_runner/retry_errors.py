"""Optional role-aware decoder retry. Base predictions and run.py are never modified."""
import argparse
import copy
import json
import os
import time
import urllib.request
from pathlib import Path

import run as base
from cqa import evaluator
from cqa.domain import digest
from cqa.quality import contract_for
from cqa.provider import ProviderError

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'source_message_enum_v7_role_aware_retry'
MODEL_DIGEST = 'ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226'


def runtime():
    result = base.runtime()
    result['base_evidence_protocol'] = result['evidence_protocol']
    result['base_adapter_source_hash'] = result['adapter_source_hash']
    result['evidence_protocol'] = PROTOCOL
    result['adapter_source_hash'] = digest(Path(__file__).read_text())
    return result


def experiment_config(provider, review=True):
    result = base.experiment_config(provider, review)
    result['prompt_version'] = evaluator.PROMPT_VERSION + '+' + PROTOCOL
    result['output_protocol'] = PROTOCOL
    result['runtime'] = runtime()
    return result


def role_aware_schema(schema, conversation, checklist):
    """Restrict only combinations already rejected by validate_proofs.

    No answer or state is selected on semantic grounds. A positive operator action
    must cite that operator; absent/uncertain checks can cite client/system context.
    """
    result = base.constrained_schema(schema, conversation)
    messages = conversation['transcript']
    first_operator = next((m['id'] for m in messages if m['speaker'] == 'operator'), None)
    criteria = result['properties']['criteria']['properties']
    for rule in checklist['criteria']:
        contract = contract_for(rule)
        if not contract: continue
        row = criteria[rule['id']]
        for branch in row.get('anyOf', [row]):
            checks = branch['properties']['checks']['properties']
            for key, spec in contract['checks'].items():
                if not spec.get('speaker') and not spec.get('first_operator'): continue
                proof = checks[key]
                nonpositive = copy.deepcopy(proof)
                nonpositive['properties']['state']['enum'] = ['absent', 'uncertain']
                eligible = [m for m in messages if m['text'].strip()
                            and (not spec.get('speaker') or m['speaker'] == spec['speaker'])
                            and (not spec.get('first_operator') or m['id'] == first_operator)]
                branches = []
                if eligible:
                    positive = copy.deepcopy(proof)
                    positive['properties']['state']['enum'] = ['present']
                    positive['properties']['evidence']['items'] = {'anyOf': [evaluator.obj({
                        'message_id': {'type': 'string', 'const': m['id']},
                        'quote': {'type': 'string', 'const': m['text']}}) for m in eligible]}
                    branches.append(positive)
                branches.append(nonpositive)
                checks[key] = {'anyOf': branches}
    return result


class RetryProvider(base.LocalProvider):
    def complete(self, system, user, schema):
        remaining = self.deadline - time.time()
        if remaining < 45: raise ProviderError('Evaluation deadline reached before retry model request')
        started = time.monotonic()
        effective_system = system + base.SYSTEM_SUFFIX
        effective_user = user + base.SCHEMA_PREFIX + json.dumps(schema, ensure_ascii=False)
        effective_schema = role_aware_schema(schema, self.conversation, self.checklist)
        body = {'model': self.cfg['model'], 'messages': [
            {'role': 'system', 'content': effective_system}, {'role': 'user', 'content': effective_user}],
            'stream': False, 'think': False, 'format': effective_schema,
            'options': {'temperature': 0, 'num_ctx': 32768, 'num_predict': 8192}}
        trace = {'conversation_id': self.conversation['conversation_id'], 'call': len(self.calls) + 1,
                 'retry_protocol': PROTOCOL, 'started_unix': time.time(),
                 'system_prompt_hash': digest(effective_system), 'user_prompt_hash': digest(effective_user),
                 'decoder_schema_hash': digest(effective_schema)}
        usage = {'input_tokens': 0, 'output_tokens': 0, 'seconds': 0, 'cost': None, 'http_attempts': 1}
        try:
            req = urllib.request.Request(self.cfg['base_url'] + '/api/chat',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=min(300, remaining - 15)) as response:
                raw = json.load(response)
            usage.update(input_tokens=raw.get('prompt_eval_count', 0), output_tokens=raw.get('eval_count', 0))
            trace['raw_content'] = raw.get('message', {}).get('content', '')
            trace['done_reason'] = raw.get('done_reason')
            if raw.get('done_reason') == 'length': raise ProviderError('Model output reached token limit')
            value = json.loads(trace['raw_content']); trace['value'] = value
            try: base.validate(value, self.checklist, self.conversation)
            except (ValueError, TypeError, KeyError, AttributeError) as err: trace['validation_error'] = str(err)
            return value, usage
        except Exception as err:
            trace['request_error'] = type(err).__name__ + ': ' + str(err)
            raise
        finally:
            usage['seconds'] = round(time.monotonic() - started, 2)
            trace['usage'] = dict(usage); self.calls.append(trace)
            base.append_trace(ROOT/'retry-trace.jsonl', trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    deadline = float(os.environ['RUN_DEADLINE_UNIX'])
    if time.time() >= deadline: raise SystemExit('RUN_DEADLINE_UNIX must be in the future')
    conversations = json.loads((ROOT/'conversations.json').read_text())
    checklist = json.loads((ROOT/'checklist.json').read_text())
    base_predictions = json.loads((ROOT/'predictions.json').read_text())
    originals = {c['conversation_id']: c for c in conversations}
    if len(base_predictions) != len(originals) or {r['conversation_id'] for r in base_predictions} != set(originals):
        raise SystemExit('Base pass must first contain one result/error for every source conversation')
    errors = {r['conversation_id']: r for r in base_predictions if r.get('error')}
    print('BASE_ERRORS', json.dumps({cid: r['error'] for cid, r in errors.items()}, ensure_ascii=False), flush=True)
    if not errors:
        print('NO_RETRIES_NEEDED', flush=True); return
    for cid, record in errors.items():
        if record.get('transcript_hash') != digest(originals[cid]) or record.get('checklist_hash') != digest(checklist):
            raise SystemExit('Base error input provenance does not match frozen inputs')
    output = ROOT/'retry-predictions.json'
    if output.exists() and not args.resume:
        raise SystemExit('Retry results already exist: use --resume to preserve attempts')
    results = json.loads(output.read_text()) if output.exists() else []
    if any(r.get('conversation_id') not in errors for r in results):
        raise SystemExit('Retry file includes a record outside the current base error set')
    provider = RetryProvider(checklist, deadline)
    with urllib.request.urlopen(provider.cfg['base_url'] + '/api/tags', timeout=20) as response:
        models = json.load(response).get('models', [])
    model = next((m for m in models if m.get('name') == provider.cfg['model']), None)
    if MODEL_DIGEST != base.MODEL_DIGEST or not model or model.get('digest') != MODEL_DIGEST:
        raise SystemExit('Expected baseline model digest not available')
    experiment = experiment_config(provider, True)
    base.atomic_json(ROOT/'retry-experiment.json', experiment)
    success = {r['conversation_id'] for r in results
               if base.reusable(r, originals[r['conversation_id']], checklist, experiment)}
    previous_config = evaluator.experiment_config
    evaluator.experiment_config = experiment_config
    try:
        for cid, previous in errors.items():
            if cid in success: continue
            if deadline - time.time() < 90:
                print('DEADLINE: preserving retries', flush=True); break
            conversation = originals[cid]; provider.conversation = conversation; provider.calls = []
            attempts = [r for r in results if r['conversation_id'] == cid]
            record = {'conversation_id': cid, 'language': conversation['language'], 'channel': conversation['channel'],
                      'transcript_hash': digest(conversation), 'checklist_hash': digest(checklist),
                      'experiment_config': experiment, 'experiment_hash': digest(experiment), 'runtime': runtime(),
                      'base_error': previous['error'], 'base_record_hash': digest(previous),
                      'base_experiment_hash': previous.get('experiment_hash'),
                      'base_all_calls_usage': previous.get('all_calls_usage'),
                      'retry_attempt': len(attempts) + 1}
            print('RETRY_START', cid, flush=True)
            try:
                result = evaluator.evaluate(conversation, checklist, provider=provider, review=True,
                    progress=lambda text: print(cid, text, flush=True))
                base.validate(result['assessment'], checklist, conversation)
                base.validate(result['draft'], checklist, conversation)
                result['prompt_version'] = experiment['prompt_version']
                result['base_prompt_version'] = evaluator.PROMPT_VERSION
                record.update(result)
                print('RETRY_DONE', cid, result['score']['total'], flush=True)
            except Exception as err:
                record['error'] = type(err).__name__ + ': ' + str(err)
                print('RETRY_FAILED', cid, record['error'], flush=True)
            record['all_calls_usage'] = {'calls': len(provider.calls),
                'input_tokens': sum(c['usage']['input_tokens'] for c in provider.calls),
                'output_tokens': sum(c['usage']['output_tokens'] for c in provider.calls),
                'seconds': round(sum(c['usage']['seconds'] for c in provider.calls), 2), 'cost': None}
            results.append(record)
            base.atomic_json(output, results)
    finally:
        evaluator.experiment_config = previous_config
    latest = {r['conversation_id']: r for r in results}
    report = {'base_errors': {cid: r['error'] for cid, r in errors.items()},
              'retry_attempts': len(results), 'retry_successes': [cid for cid, r in latest.items() if not r.get('error')],
              'retry_failures': {cid: r['error'] for cid, r in latest.items() if r.get('error')},
              'not_attempted': sorted(set(errors) - set(latest)),
              'base_predictions_modified': False, 'retry_protocol': PROTOCOL}
    base.atomic_json(ROOT/'retry-report.json', report)
    print('RETRY_FINISHED', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
