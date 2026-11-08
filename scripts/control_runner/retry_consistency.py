"""Optional closing-consistency retry; v6/v7 artifacts and the app stay unchanged."""
import argparse
import copy
import json
import os
import time
import urllib.request
from pathlib import Path

import run as base
import retry_errors as roles
from cqa import evaluator
from cqa.domain import digest
from cqa.quality import contract_for
from cqa.provider import ProviderError

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'source_message_enum_v8_closing_consistency'
MODEL_DIGEST = 'ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226'


def runtime():
    result = base.runtime()
    result['base_evidence_protocol'] = result['evidence_protocol']
    result['base_adapter_source_hash'] = result['adapter_source_hash']
    result['evidence_protocol'] = PROTOCOL
    result['adapter_source_hash'] = digest(Path(__file__).read_text())
    result['role_adapter_source_hash'] = digest((ROOT/'retry_errors.py').read_text())
    result['role_evidence_protocol'] = roles.PROTOCOL
    return result


def experiment_config(provider, review=True):
    result = base.experiment_config(provider, review)
    result['prompt_version'] = evaluator.PROMPT_VERSION + '+' + PROTOCOL
    result['output_protocol'] = PROTOCOL
    result['runtime'] = runtime()
    return result


def restrict_states(proof_schema, allowed):
    """Intersect state enums through nested anyOf while keeping evidence bounds."""
    node = copy.deepcopy(proof_schema)
    if 'anyOf' in node:
        parts = [restrict_states(part, allowed) for part in node['anyOf']]
        node['anyOf'] = [part for part in parts if part is not None]
        return node if node['anyOf'] else None
    state = node.get('properties', {}).get('state')
    if state is None: raise ValueError('Proof schema has no state')
    state['enum'] = [value for value in state['enum'] if value in allowed]
    return node if state['enum'] else None


def consistent_schema(schema, conversation, checklist):
    result = roles.role_aware_schema(schema, conversation, checklist)
    criteria = result['properties']['criteria']['properties']
    for rule in checklist['criteria']:
        contract = contract_for(rule)
        if not contract or contract['aggregation'] not in ['closing', 'identity']: continue
        rows = criteria[rule['id']].get('anyOf', [criteria[rule['id']]])
        # Five exhaustive, mutually exclusive truth branches. State facts are
        # still chosen by the model; only impossible fact/answer combinations go.
        branch_specs = [
            ('yes', {'summary_or_help': ['present'], 'farewell': ['present']}),
            ('na', {'opportunity': ['absent'], 'summary_or_help': ['absent', 'uncertain']}),
            ('na', {'opportunity': ['absent'], 'summary_or_help': ['present'], 'farewell': ['absent', 'uncertain']}),
            ('no', {'opportunity': ['present', 'uncertain'], 'summary_or_help': ['absent', 'uncertain']}),
            ('no', {'opportunity': ['present', 'uncertain'], 'summary_or_help': ['present'], 'farewell': ['absent', 'uncertain']})]
        if contract['aggregation'] == 'identity':
            branch_specs = [
                ('na', {'protected_action': ['absent']}),
                ('yes', {'protected_action': ['present'], 'verified_before_action': ['present']}),
                ('no', {'protected_action': ['present'], 'verified_before_action': ['absent', 'uncertain']}),
                ('no', {'protected_action': ['uncertain']})]
        output = []
        for answer, limits in branch_specs:
            for source in rows:
                if answer not in source['properties']['answer']['enum']: continue
                branch = copy.deepcopy(source)
                branch['properties']['answer']['enum'] = [answer]
                proofs = branch['properties']['checks']['properties']
                valid = True
                for key, allowed in limits.items():
                    proofs[key] = restrict_states(proofs[key], allowed)
                    if proofs[key] is None: valid = False; break
                if valid: output.append(branch)
        if not output: raise ValueError('Proof schema has no satisfiable branches')
        criteria[rule['id']] = {'anyOf': output}
    return result


class ConsistencyProvider(base.LocalProvider):
    def complete(self, system, user, schema):
        remaining = self.deadline - time.time()
        if remaining < 45: raise ProviderError('Evaluation deadline reached before retry model request')
        started = time.monotonic()
        effective_system = system + base.SYSTEM_SUFFIX
        effective_user = user + base.SCHEMA_PREFIX + json.dumps(schema, ensure_ascii=False)
        effective_schema = consistent_schema(schema, self.conversation, self.checklist)
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
            base.append_trace(ROOT/'consistency-trace.jsonl', trace)


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
    role_path = ROOT/'retry-predictions.json'
    if not role_path.exists(): raise SystemExit('Run v7 role-aware retries before consistency retries')
    role_records = json.loads(role_path.read_text())
    if any(r.get('conversation_id') not in errors for r in role_records):
        raise SystemExit('V7 retry contains a record outside base errors')
    latest_roles = {r['conversation_id']: r for r in role_records}
    if set(latest_roles) != set(errors):
        raise SystemExit('Every base error needs a v7 attempt before consistency retries')
    role_provider = roles.RetryProvider(checklist, deadline)
    role_experiment = roles.experiment_config(role_provider, True)
    for cid, record in latest_roles.items():
        if (record.get('experiment_config') != role_experiment
                or record.get('experiment_hash') != digest(role_experiment)
                or record.get('transcript_hash') != digest(originals[cid])
                or record.get('checklist_hash') != digest(checklist)
                or record.get('base_record_hash') != digest(errors[cid])):
            raise SystemExit('Latest v7 provenance does not match frozen base inputs/adapters')
        if not record.get('error') and not base.reusable(record, originals[cid], checklist, role_experiment):
            raise SystemExit('Latest v7 success failed validation')
    errors = {cid: record for cid, record in errors.items() if latest_roles[cid].get('error')}
    if not errors:
        print('V7_RESOLVED_ALL_ERRORS', flush=True); return
    output = ROOT/'consistency-predictions.json'
    if output.exists() and not args.resume:
        raise SystemExit('Retry results already exist: use --resume to preserve attempts')
    results = json.loads(output.read_text()) if output.exists() else []
    if any(r.get('conversation_id') not in errors for r in results):
        raise SystemExit('Retry file includes a record outside the current base error set')
    provider = ConsistencyProvider(checklist, deadline)
    with urllib.request.urlopen(provider.cfg['base_url'] + '/api/tags', timeout=20) as response:
        models = json.load(response).get('models', [])
    model = next((m for m in models if m.get('name') == provider.cfg['model']), None)
    if MODEL_DIGEST != base.MODEL_DIGEST or not model or model.get('digest') != MODEL_DIGEST:
        raise SystemExit('Expected baseline model digest not available')
    experiment = experiment_config(provider, True)
    base.atomic_json(ROOT/'consistency-experiment.json', experiment)
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
                      'retry_attempt': len(attempts) + 1,
                      'prior_retry_error': latest_roles[cid]['error'],
                      'prior_retry_record_hash': digest(latest_roles[cid]),
                      'prior_retry_experiment_hash': latest_roles[cid].get('experiment_hash'),
                      'prior_retry_all_calls_usage': latest_roles[cid].get('all_calls_usage')}
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
    base.atomic_json(ROOT/'consistency-report.json', report)
    print('RETRY_FINISHED', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
