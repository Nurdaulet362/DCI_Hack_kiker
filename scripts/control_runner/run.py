"""Budget-bounded, auditable Qwen rerun. No network calls occur on import."""
import argparse
import copy
import json
import os
import re
import time
import urllib.request
from pathlib import Path

os.environ.update(AI_PROVIDER='ollama', AI_BASE_URL='http://127.0.0.1:11434',
                  AI_MODEL='qwen3:30b', AI_REVIEW='true', AI_MAX_OUTPUT_TOKENS='8192',
                  AI_TIMEOUT_SECONDS='300', AI_REASONING_EFFORT='', AI_API_KEY='',
                  AI_CONTEXT_WINDOW='32768', AI_THINK='false')
from cqa import evaluator
from cqa.domain import validate_assessment, calculate, digest
from cqa.quality import validate_proofs
from cqa.provider import Provider, ProviderError

ROOT = Path(__file__).resolve().parent
MODEL_DIGEST = 'ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226'
PROTOCOL = 'source_message_enum_v6_facts_first'
SYSTEM_SUFFIX = ('\nДля данного протокола полная исходная реплика имеет приоритет над рекомендацией сокращать цитаты. Для каждого evidence, в том числе внутри checks, выбирай исходную реплику '
    'целиком и её id: quote точно совпадает с полным text этой реплики. '
    'Нельзя переписывать, сокращать или перефразировать цитаты. '
    'confidence — число от 0 до 1. Проверь весь диалог; checked_start и checked_end — '
    'первая и последняя реплики диалога. Коды answer: yes = Да, no = Нет, na = Н/П. '
    'Если выполнено na_rule и allow_na=true, выбери na, а не no. '
    'Отсутствие действия, которое не требовалось, не является нарушением. '
    'Числовые ступени передавай строкой, как в enum. reason и summary — '
    'содержательные объяснения на русском языке, не ID реплик. '
    'Сначала заполни checks, затем выведи answer строго по answer_from_checks. '
    'Не меняй факты ради сохранения итогового ответа. Если enough_initial=present, '
    'то по соответствующему контракту answer=na, независимо от clarified_before_answer.')
SCHEMA_PREFIX = '\nJSON_SCHEMA (формат ответа, не данные диалога):\n'
BASE_EXPERIMENT = evaluator.experiment_config


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.flush(); os.fsync(file.fileno())
    temporary.replace(path)


def append_trace(path, value):
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'a') as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def runtime():
    return {'think': False, 'num_ctx': 32768, 'num_predict': 8192, 'temperature': 0,
            'evidence_protocol': PROTOCOL, 'quantization': 'Q4_K_M',
            'model_digest': MODEL_DIGEST, 'adapter_source_hash': digest(Path(__file__).read_text()),
            'evaluator_source_hash': digest((ROOT/'cqa/evaluator.py').read_text()),
            'quality_source_hash': digest((ROOT/'cqa/quality.py').read_text())}


def experiment_config(provider, review=True):
    config = BASE_EXPERIMENT(provider, review)
    config.update(base_prompt_version=config['prompt_version'],
                  base_prompt_hash=config['prompt_hash'],
                  prompt_version=config['prompt_version'] + '+' + PROTOCOL,
                  prompt_hash=digest(evaluator.SYSTEM + SYSTEM_SUFFIX),
                  schema_prompt_hash=digest(SCHEMA_PREFIX + json.dumps(provider.base_schema, ensure_ascii=False)),
                  output_protocol=PROTOCOL, runtime=runtime())
    return config


evaluator.experiment_config = experiment_config


def constrained_schema(schema, conversation):
    schema = copy.deepcopy(schema)
    messages = conversation['transcript']
    evidence_choices = [evaluator.obj({'message_id': {'type': 'string', 'const': m['id']},
                                      'quote': {'type': 'string', 'const': m['text']}})
                        for m in messages if m['text'].strip()]
    if not evidence_choices:
        raise ProviderError('Transcript contains no nonempty evidence text')

    def constrain(node):
        if not isinstance(node, dict): return
        props = node.get('properties', {})
        if 'evidence' in props:
            props['evidence'] = {'type': 'array', 'items': {'anyOf': copy.deepcopy(evidence_choices)},
                                 'minItems': 1, 'maxItems': 3}
        if 'checked_start' in props:
            props['checked_start'] = {'type': 'string', 'const': messages[0]['id']}
            props['checked_end'] = {'type': 'string', 'const': messages[-1]['id']}
        if 'confidence' in props:
            props['confidence'] = {'type': 'number', 'minimum': 0, 'maximum': 1}
        for key, value in props.items():
            if key != 'evidence': constrain(value)
    constrain(schema)
    criteria = schema['properties']['criteria']['properties']
    for cid, row in list(criteria.items()):
        fields = row['properties']
        if 'na' in fields['answer']['enum']:
            applicable, not_applicable = copy.deepcopy(row), copy.deepcopy(row)
            applicable['properties']['answer']['enum'] = [x for x in fields['answer']['enum'] if x != 'na']
            applicable['properties']['support_type']['enum'] = ['observed', 'absence']
            not_applicable['properties']['answer']['enum'] = ['na']
            not_applicable['properties']['support_type']['enum'] = ['not_applicable']
            criteria[cid] = {'anyOf': [applicable, not_applicable]}
        else:
            fields['support_type']['enum'] = ['observed', 'absence']
    return schema


def reject_placeholders(value):
    texts = [value.get('summary', '')]
    for row in value.get('criteria', {}).values():
        texts.append(row.get('reason', ''))
        texts.extend(proof.get('reason', '') for proof in row.get('checks', {}).values())
    if any(not isinstance(t, str) or not t.strip() or re.fullmatch(r'm\d+(?:\s*[,;–-]\s*m\d+)*', t.strip()) for t in texts):
        raise ValueError('Placeholder explanation detected')


def validate(value, checklist, conversation):
    validate_proofs(validate_assessment(value, checklist, conversation), checklist, conversation)
    reject_placeholders(value)
    return value


class LocalProvider(Provider):
    def __init__(self, checklist, deadline):
        super().__init__()
        self.checklist, self.deadline = checklist, deadline
        self.base_schema = evaluator.response_schema(checklist)
        self.conversation = None
        self.calls = []

    def complete(self, system, user, schema):
        remaining = self.deadline - time.time()
        if remaining < 45:
            raise ProviderError('Evaluation deadline reached before another model request')
        started = time.monotonic()
        effective_system = system + SYSTEM_SUFFIX
        effective_user = user + SCHEMA_PREFIX + json.dumps(schema, ensure_ascii=False)
        effective_schema = constrained_schema(schema, self.conversation)
        body = {'model': self.cfg['model'], 'messages': [
            {'role': 'system', 'content': effective_system}, {'role': 'user', 'content': effective_user}],
            'stream': False, 'think': False, 'format': effective_schema,
            'options': {'temperature': 0, 'num_ctx': 32768, 'num_predict': 8192}}
        trace = {'conversation_id': self.conversation['conversation_id'], 'call': len(self.calls) + 1,
                 'started_unix': time.time(), 'system_prompt_hash': digest(effective_system),
                 'user_prompt_hash': digest(effective_user), 'decoder_schema_hash': digest(effective_schema)}
        usage = {'input_tokens': 0, 'output_tokens': 0, 'seconds': 0, 'cost': None, 'http_attempts': 1}
        try:
            req = urllib.request.Request(self.cfg['base_url'] + '/api/chat',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=min(300, remaining - 15)) as response:
                raw = json.load(response)
            usage.update(input_tokens=raw.get('prompt_eval_count', 0), output_tokens=raw.get('eval_count', 0))
            trace['raw_content'] = raw.get('message', {}).get('content', '')
            trace['done_reason'] = raw.get('done_reason')
            if raw.get('done_reason') == 'length':
                raise ProviderError('Model output reached token limit')
            value = json.loads(trace['raw_content'])
            trace['value'] = value
            try: validate(value, self.checklist, self.conversation)
            except (ValueError, TypeError, KeyError, AttributeError) as err:
                trace['validation_error'] = str(err)
            return value, usage
        except Exception as err:
            trace['request_error'] = type(err).__name__ + ': ' + str(err)
            raise
        finally:
            usage['seconds'] = round(time.monotonic() - started, 2)
            trace['usage'] = dict(usage)
            self.calls.append(trace)
            append_trace(ROOT/'trace.jsonl', trace)


def reusable(record, conversation, checklist, experiment):
    if (record.get('error') or record.get('transcript_hash') != digest(conversation)
            or record.get('checklist_hash') != digest(checklist)
            or record.get('experiment_hash') != digest(experiment)
            or record.get('experiment_config') != experiment): return False
    try:
        validate(record['assessment'], checklist, conversation)
        validate(record['draft'], checklist, conversation)
        return record['score'] == calculate(checklist, record['assessment'])
    except (ValueError, TypeError, KeyError, AttributeError): return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    deadline = float(os.environ['RUN_DEADLINE_UNIX'])
    if not time.time() < deadline: raise SystemExit('RUN_DEADLINE_UNIX must be in the future')
    conversations = json.loads((ROOT/'conversations.json').read_text())
    checklist = json.loads((ROOT/'checklist.json').read_text())
    provider = LocalProvider(checklist, deadline)
    with urllib.request.urlopen(provider.cfg['base_url'] + '/api/tags', timeout=20) as response:
        models = json.load(response).get('models', [])
    model = next((m for m in models if m.get('name') == provider.cfg['model']), None)
    if not model or model.get('digest') != MODEL_DIGEST:
        raise SystemExit('Expected baseline model digest not available; refusing incomparable run')
    experiment = experiment_config(provider, True)
    atomic_json(ROOT/'experiment.json', experiment)
    output = ROOT/'predictions.json'
    results = json.loads(output.read_text()) if args.resume and output.exists() else []
    by_id = {c['conversation_id']: c for c in conversations}
    results = [r for r in results if r.get('conversation_id') in by_id]
    success = {r['conversation_id'] for r in results if reusable(r, by_id[r['conversation_id']], checklist, experiment)}
    pending = [c for c in conversations if c['conversation_id'] not in success]
    if args.limit is not None:
        if args.limit < 1: raise SystemExit('--limit must be positive')
        pending = pending[:args.limit]
    for conversation in pending:
        if deadline - time.time() < 90:
            print('DEADLINE: preserving completed records', flush=True); break
        cid = conversation['conversation_id']; provider.conversation = conversation; provider.calls = []
        print('START', cid, flush=True)
        record = {'conversation_id': cid, 'language': conversation['language'], 'channel': conversation['channel'],
                  'transcript_hash': digest(conversation), 'checklist_hash': digest(checklist),
                  'experiment_config': experiment, 'experiment_hash': digest(experiment), 'runtime': runtime()}
        try:
            result = evaluator.evaluate(conversation, checklist, provider=provider, review=True,
                progress=lambda stage: print(cid, stage, flush=True))
            validate(result['assessment'], checklist, conversation)
            validate(result['draft'], checklist, conversation)
            result['prompt_version'] = experiment['prompt_version']
            result['base_prompt_version'] = evaluator.PROMPT_VERSION
            record.update(result)
            print('DONE', cid, result['score']['total'], flush=True)
        except Exception as err:
            record['error'] = type(err).__name__ + ': ' + str(err)
            print('FAILED', cid, record['error'], flush=True)
        record['all_calls_usage'] = {'calls': len(provider.calls),
            'input_tokens': sum(c['usage']['input_tokens'] for c in provider.calls),
            'output_tokens': sum(c['usage']['output_tokens'] for c in provider.calls),
            'seconds': round(sum(c['usage']['seconds'] for c in provider.calls), 2), 'cost': None}
        results = [r for r in results if r['conversation_id'] != cid] + [record]
        atomic_json(output, results)
    print('FINISHED', len(results), 'successful', sum(not r.get('error') for r in results), flush=True)


if __name__ == '__main__': main()
