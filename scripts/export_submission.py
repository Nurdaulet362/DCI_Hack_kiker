#!/usr/bin/env python3
"""Export actual CQA evaluations to organizer JSON 1.0; never invent answers.

Source input is the original API payload or a list of original conversations.
Error/missing records remain explicit status=error. Run with --allow-errors only
when deliberately submitting incomplete processing before the deadline.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

IDS = {
    'contact': 'contact', 'needs': 'discovery', 'identity': 'identification',
    'consultation': 'consultation', 'empathy': 'empathy',
    'management': 'dialogue', 'closing': 'closure', 'gross_error': 'critical',
}
WEIGHTS = {'contact': 10, 'discovery': 15, 'identification': 20,
           'consultation': 20, 'empathy': 15, 'dialogue': 15, 'closure': 5, 'critical': 0}
SCALES = {'consultation': {0, 5, 10, 15, 20}, 'dialogue': {0, 4, 7, 10}}
NULLABLE = {'discovery', 'identification', 'consultation', 'closure'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_records(payload):
    if isinstance(payload, list): rows = payload
    elif isinstance(payload, dict):
        if payload.get('isSuccess') is False: raise ValueError('Dataset API reported isSuccess=false')
        data = payload.get('data', payload)
        if not isinstance(data, dict): raise ValueError('Dataset data must be an object')
        rows = data.get('records')
        if rows is None and 'conversation_id' in data: rows = [data]
        if 'total_records' in data and isinstance(rows, list) and data['total_records'] != len(rows):
            raise ValueError('Dataset total_records does not match array length')
    else: rows = None
    if not isinstance(rows, list) or not rows: raise ValueError('Source needs nonempty conversations array')
    ids = [r.get('conversation_id') for r in rows if isinstance(r, dict)]
    if len(ids) != len(rows) or any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Source has missing or duplicate conversation_id')
    return rows


def safe_error(message):
    text = str(message)
    text = re.sub(r'https?://\S+', '[URL removed]', text)
    text = re.sub(r'(?:sk-|bak-)[A-Za-z0-9_-]+|AQ\.[A-Za-z0-9_.-]+', '[credential removed]', text)
    return text[:1000] or 'Не удалось получить корректную оценку модели.'


def error_record(cid, message):
    return {'conversation_id': cid, 'status': 'error', 'criteria': [], 'score': None, 'error': safe_error(message)}


def actual_value(answer, public_id):
    if answer == 'na':
        if public_id not in NULLABLE: raise ValueError(f'{public_id}: N/A is forbidden')
        return None
    if public_id in SCALES:
        # Exact scale strings only, no inferred rounding or clipping.
        if answer not in {str(n) for n in SCALES[public_id]}: raise ValueError(f'{public_id}: invalid scale')
        return int(answer)
    if answer not in ['yes', 'no']: raise ValueError(f'{public_id}: invalid boolean answer')
    return answer == 'yes'


def score_from_values(rows):
    denominator = Decimal(0); numerator = Decimal(0)
    for row in rows:
        cid, value = row['criterion_id'], row['value']
        if value is None: continue
        weight = Decimal(WEIGHTS[cid]); denominator += weight
        if cid in SCALES: ratio = Decimal(value) / Decimal(max(SCALES[cid]))
        else: ratio = Decimal(int(value))
        numerator += weight * ratio
    if not denominator: raise ValueError('No applicable weighted criteria')
    total = float((100 * numerator / denominator).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    return {'final_score': total,
            'critical_flag': next(r['value'] for r in rows if r['criterion_id'] == 'critical')}


def export_one(record, raw, checklist, domain):
    cid = raw['conversation_id']
    if record is None: return error_record(cid, 'Оценка модели не получена к моменту экспорта.')
    if record.get('error') or record.get('status') == 'error':
        return error_record(cid, record.get('error') or 'Модель не смогла оценить диалог.')
    try:
        conv = domain.normalize_conversation(raw)
        if record.get('transcript_hash') != domain.digest(conv): raise ValueError('Transcript hash mismatch or missing')
        if record.get('checklist_hash') != domain.digest(checklist): raise ValueError('Checklist hash mismatch or missing')
        if not isinstance(record.get('model'), str) or not record['model'].strip(): raise ValueError('Actual model name is missing')
        assessment = domain.validate_assessment(record.get('assessment'), checklist, conv)
        calculated = domain.calculate(checklist, assessment)
        stored = record.get('score')
        if not isinstance(stored, dict) or stored.get('total') != calculated['total'] or stored.get('gross_error') != calculated['gross_error']:
            raise ValueError('Stored score contradicts criterion answers')
        turns = {m['id']: (i+1, m['text']) for i, m in enumerate(conv['transcript'])}
        criteria = []
        for internal, public in IDS.items():
            row = assessment['criteria'][internal]
            # All citations are checked against source before converting to original turn indices.
            evidence = list(row['evidence'])
            for proof in row.get('checks', {}).values(): evidence += proof.get('evidence', [])
            indices = set()
            for item in evidence:
                mid, quote = item.get('message_id'), item.get('quote')
                if mid not in turns or not isinstance(quote, str) or not quote.strip() or quote not in turns[mid][1]:
                    raise ValueError(f'{internal}: unsupported evidence')
                indices.add(turns[mid][0])
            criteria.append({'criterion_id': public, 'value': actual_value(row['answer'], public),
                             'rationale': row['reason'].strip(), 'evidence_turns': sorted(indices)})
        score = score_from_values(criteria)
        if score['final_score'] != calculated['total'] or score['critical_flag'] != calculated['gross_error']:
            raise ValueError('Organizer weights/scales disagree with application calculation')
        return {'conversation_id': cid, 'status': 'ok', 'criteria': criteria, 'score': score}
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        return error_record(cid, f'Результат не прошёл проверку экспорта: {exc}')


def validate_submission(payload, raw_records, expected_count):
    if set(payload) != {'schema_version', 'team_name', 'model', 'results'}: raise ValueError('Incorrect root fields')
    if payload['schema_version'] != '1.0': raise ValueError('Incorrect schema_version')
    if not isinstance(payload['team_name'], str) or not payload['team_name'].strip(): raise ValueError('Missing team_name')
    model = payload['model']
    if set(model) != {'mode', 'name'} or model['mode'] not in ['local', 'external_api', 'hybrid'] or not isinstance(model['name'], str) or not model['name'].strip():
        raise ValueError('Invalid model metadata')
    results = payload['results']
    if not isinstance(results, list) or len(results) != expected_count or len(raw_records) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} source and output records')
    sources = {r['conversation_id']: r for r in raw_records}
    ids = [r['conversation_id'] for r in results]
    if len(set(ids)) != len(ids) or set(ids) != set(sources): raise ValueError('Output/source IDs mismatch')
    for result in results:
        cid = result['conversation_id']
        if result['status'] == 'error':
            if set(result) != {'conversation_id','status','criteria','score','error'} or result['criteria'] != [] or result['score'] is not None or not isinstance(result['error'], str) or not result['error'].strip():
                raise ValueError(f'{cid}: invalid error record')
            continue
        if result['status'] != 'ok' or set(result) != {'conversation_id','status','criteria','score'}: raise ValueError(f'{cid}: invalid success record')
        criteria = result['criteria']
        if len(criteria) != 8 or {r['criterion_id'] for r in criteria} != set(IDS.values()): raise ValueError(f'{cid}: need exactly 8 distinct criteria')
        for row in criteria:
            if set(row) != {'criterion_id','value','rationale','evidence_turns'}: raise ValueError(f'{cid}: invalid criterion fields')
            key, value = row['criterion_id'], row['value']
            if value is None:
                if key not in NULLABLE: raise ValueError(f'{cid}/{key}: null forbidden')
            elif key in SCALES:
                if type(value) is not int or value not in SCALES[key]: raise ValueError(f'{cid}/{key}: invalid numeric scale')
            elif type(value) is not bool: raise ValueError(f'{cid}/{key}: need JSON boolean')
            if not isinstance(row['rationale'], str) or not row['rationale'].strip(): raise ValueError(f'{cid}/{key}: empty rationale')
            turns = row['evidence_turns']
            if not isinstance(turns, list) or not turns or len(set(turns)) != len(turns) or any(type(n) is not int or not 1 <= n <= len(sources[cid]['transcript']) for n in turns):
                raise ValueError(f'{cid}/{key}: invalid original turn indices')
        score = result['score']
        if not isinstance(score, dict) or set(score) != {'final_score','critical_flag'} or type(score['final_score']) not in [int,float] or not math.isfinite(score['final_score']) or not 0 <= score['final_score'] <= 100 or type(score['critical_flag']) is not bool:
            raise ValueError(f'{cid}: invalid score')
        if score != score_from_values(criteria): raise ValueError(f'{cid}: final_score or critical_flag inconsistent')
    return {'records': len(results), 'ok': sum(r['status']=='ok' for r in results), 'errors': sum(r['status']=='error' for r in results)}


def write_atomic(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.submission-', delete=False) as stream:
            temp = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp and temp.exists(): temp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--checklist', type=Path)
    parser.add_argument('--team-name', default='DCI_Hack_kiker')
    parser.add_argument('--model-mode', choices=['local','external_api','hybrid'])
    parser.add_argument('--model-name', help='Actual used model(s); inferred from successful evaluation metadata if omitted')
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate', type=Path, help='Validate existing submission instead of exporting')
    parser.add_argument('--allow-errors', action='store_true')
    args = parser.parse_args()
    raw = source_records(load(args.source))
    if args.validate:
        payload = load(args.validate)
    else:
        if not all([args.predictions,args.checklist,args.output,args.model_mode]):
            parser.error('Export requires --predictions --checklist --output --model-mode')
        sys.path.insert(0, str(args.project))
        from cqa import domain
        checklist = domain.validate_checklist(load(args.checklist))
        if {c['id'] for c in checklist['criteria']} != set(IDS): raise ValueError('Submission requires the original eight criterion IDs')
        for c in checklist['criteria']:
            if c['weight'] != WEIGHTS[IDS[c['id']]]: raise ValueError('Checklist weights differ from organizer format')
        predictions = load(args.predictions)
        if not isinstance(predictions, list): raise ValueError('Predictions must be a list of actual CQA evaluation records')
        lookup = {}
        for p in predictions:
            cid = p.get('conversation_id')
            if cid in lookup: raise ValueError('Duplicate prediction ID')
            lookup[cid] = p
        wanted = {r['conversation_id'] for r in raw}
        if set(lookup)-wanted: raise ValueError('Predictions contain IDs outside this dataset; refusing mixed/SWAP submission')
        names = sorted({p['model'] for p in predictions if isinstance(p.get('model'),str) and p['model'].strip()})
        name = args.model_name or ', '.join(names)
        if not name: raise ValueError('No actual model name; supply --model-name for a failed attempted run')
        if args.model_name and any(n not in name for n in names): raise ValueError('Provided model-name does not include actual prediction model metadata')
        payload = {'schema_version':'1.0','team_name':args.team_name,
                   'model':{'mode':args.model_mode,'name':name},
                   'results':[export_one(lookup.get(r['conversation_id']), r, checklist, domain) for r in raw]}
    stats = validate_submission(payload, raw, args.count)
    if not args.validate: write_atomic(args.output, payload)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()
    print(json.dumps({**stats, 'path':str(args.validate or args.output), 'content_sha256':hashlib.sha256(encoded).hexdigest()}, ensure_ascii=False))
    if stats['errors'] and not args.allow_errors: return 2
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, OSError, TypeError, KeyError) as error:
        print('Export validation failed: '+safe_error(error), file=sys.stderr); raise SystemExit(1)
