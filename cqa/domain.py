import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def validate_checklist(checklist):
    def text(value, label, limit, required=True):
        if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
            raise ValueError(f'{label}: нужна строка длиной до {limit} символов.')

    def number(value):
        return type(value) in [int, float] and 0 <= value <= 10000 and math.isfinite(value)

    if not isinstance(checklist, dict):
        raise ValueError('Чек-лист должен быть объектом.')
    text(checklist.get('name'), 'Название чек-листа', 200)
    criteria = checklist.get('criteria')
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= 40:
        raise ValueError('Чек-лист должен содержать от 1 до 40 критериев.')
    seen = set()
    for c in criteria:
        if not isinstance(c, dict):
            raise ValueError('Каждый критерий должен быть объектом.')
        cid = c.get('id')
        text(cid, 'ID критерия', 64)
        if cid in seen:
            raise ValueError('ID критериев должны быть уникальными.')
        seen.add(cid)
        if c.get('kind') not in ['binary', 'scale', 'flag']:
            raise ValueError('Тип критерия: binary, scale или flag.')
        if not isinstance(c.get('allow_na'), bool):
            raise ValueError('allow_na должен быть логическим значением.')
        text(c.get('na_rule', ''), 'Условие N/A', 10000, c['allow_na'])
        for key, limit in [('title', 300), ('rule', 10000), ('section', 200)]:
            text(c.get(key), key, limit)
        weight = c.get('weight')
        if not number(weight):
            raise ValueError('Вес должен быть конечным числом от 0 до 10000.')
        if c['kind'] == 'flag' and weight != 0:
            raise ValueError('Флаг нарушения должен иметь нулевой вес.')
        levels = c.get('levels', [])
        if not isinstance(levels, list) or len(levels) > 20:
            raise ValueError('Шкала должна быть списком не более чем из 20 ступеней.')
        if c['kind'] == 'scale':
            if len(levels) < 2 or any(not isinstance(level, dict) for level in levels):
                raise ValueError('Задайте от 2 до 20 ступеней шкалы в виде объектов.')
            values = [level.get('value') for level in levels]
            if not all(number(value) for value in values):
                raise ValueError('Ступени должны быть конечными числами от 0 до 10000.')
            if len(set(values)) != len(values) or min(values) != 0 or max(values) <= 0:
                raise ValueError('Шкала должна включать 0 и уникальные положительные ступени.')
            for level in levels:
                text(level.get('description'), 'Описание ступени', 5000)
        elif levels:
            raise ValueError('Ступени задаются только для критерия типа scale.')
    return checklist


def choices(c):
    values = [str(l['value']) for l in c['levels']] if c['kind'] == 'scale' else ['yes', 'no']
    return values + (['na'] if c['allow_na'] else [])


def normalize_conversation(record):
    if not isinstance(record, dict):
        raise ValueError('Диалог должен быть объектом.')
    cid = record.get('conversation_id')
    if not isinstance(cid, str) or not cid.strip() or len(cid) > 200:
        raise ValueError('Нужен conversation_id длиной до 200 символов.')
    if record.get('language') not in ['RU', 'KZ', 'EN']:
        raise ValueError('language: RU, KZ или EN.')
    if record.get('channel') not in ['agent chat', 'voice call']:
        raise ValueError('channel: agent chat или voice call.')
    transcript = record.get('transcript')
    if not isinstance(transcript, list) or not 1 <= len(transcript) <= 5000:
        raise ValueError('Транскрипт должен содержать 1–5000 реплик.')
    messages = []
    for i, m in enumerate(transcript):
        if not isinstance(m, dict) or m.get('speaker') not in ['operator', 'client', 'bot', 'system']:
            raise ValueError('Неизвестная роль в транскрипте.')
        if not isinstance(m.get('text'), str) or not isinstance(m.get('timestamp'), str):
            raise ValueError('Каждая реплика должна содержать text и исходный timestamp.')
        messages.append({'id': f'm{i+1:04d}', 'timestamp': m['timestamp'],
                         'speaker': m['speaker'], 'text': m['text']})
    if sum(len(m['text']) for m in messages) > 500000:
        raise ValueError('Транскрипт слишком большой (максимум 500 000 символов).')
    audio = record.get('audio_url')
    if audio is not None and not isinstance(audio, str):
        raise ValueError('audio_url должен быть строкой или null.')
    return {'conversation_id': cid, 'language': record['language'], 'channel': record['channel'],
            'transcript': messages, 'audio_url': audio or None}


def validate_assessment(result, checklist, conversation):
    expected = {c['id'] for c in checklist['criteria']}
    if (not isinstance(result, dict) or not isinstance(result.get('criteria'), dict)
            or set(result['criteria']) != expected):
        raise ValueError('AI должен оценить каждый критерий ровно один раз.')
    if not isinstance(result.get('summary'), str) or not isinstance(result.get('uncertainties'), list) or any(
            not isinstance(u, str) for u in result['uncertainties']):
        raise ValueError('Неверный формат резюме или неопределённостей.')
    messages = {m['id']: m for m in conversation['transcript']}
    order = {mid: i for i, mid in enumerate(messages)}
    for c in checklist['criteria']:
        r = result['criteria'][c['id']]
        if not isinstance(r, dict):
            raise ValueError('Оценка критерия должна быть объектом.')
        if r.get('answer') not in choices(c):
            raise ValueError(f"Недопустимая оценка {c['id']}.")
        if not isinstance(r.get('reason'), str) or not r['reason'].strip():
            raise ValueError('Нужно обоснование каждой оценки.')
        if type(r.get('confidence')) not in [int, float] or not 0 <= r['confidence'] <= 1:
            raise ValueError('confidence должен находиться в диапазоне 0–1.')
        if r.get('support_type') not in ['observed', 'absence', 'not_applicable']:
            raise ValueError('Неверный тип доказательства.')
        if (r['answer'] == 'na') != (r['support_type'] == 'not_applicable'):
            raise ValueError('N/A требует отдельного обоснования неприменимости.')
        start, end = r.get('checked_start'), r.get('checked_end')
        if start not in order or end not in order or order[start] > order[end]:
            raise ValueError('Неверный проверенный участок разговора.')
        evidence = r.get('evidence')
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 10:
            raise ValueError('Нужно 1–10 ссылок на доказательства или проверенный участок.')
        for e in evidence:
            if not isinstance(e, dict):
                raise ValueError('Доказательство должно быть объектом.')
            mid, quote = e.get('message_id'), e.get('quote')
            if mid not in messages or not isinstance(quote, str) or not quote.strip():
                raise ValueError('Доказательство ссылается на отсутствующую реплику.')
            if quote not in messages[mid]['text']:
                raise ValueError(f'Цитата не найдена в исходной реплике {mid}.')
            if not order[start] <= order[mid] <= order[end]:
                raise ValueError('Доказательство находится за пределами проверенного участка.')
    return result


def rounded(x):
    return float(x.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def calculate(checklist, assessment):
    # Keep Decimal precision until final presentation; flags never modify the total.
    applicable = sum((Decimal(str(c['weight'])) for c in checklist['criteria']
                      if assessment['criteria'][c['id']]['answer'] != 'na'), Decimal(0))
    earned, sections, rows, flags = Decimal(0), {}, [], []
    for c in checklist['criteria']:
        answer = assessment['criteria'][c['id']]['answer']
        weight = Decimal(str(c['weight']))
        if c['kind'] == 'flag':
            if answer == 'yes': flags.append(c['id'])
        ratio = Decimal(0)
        if answer not in ['no', 'na']:
            ratio = (Decimal(answer) / Decimal(str(max(l['value'] for l in c['levels'])))
                     if c['kind'] == 'scale' else Decimal(1))
        points = weight * ratio
        earned += points
        effective = weight * 100 / applicable if applicable and answer != 'na' else Decimal(0)
        rows.append({'id': c['id'], 'answer': answer, 'weight': c['weight'],
                     'effective_weight': rounded(effective), 'points': rounded(effective * ratio)})
        s = sections.setdefault(c['section'], {'earned': Decimal(0), 'max': Decimal(0)})
        if answer != 'na':
            s['earned'] += points
            s['max'] += weight
    return {'total': rounded(earned * 100 / applicable) if applicable else None,
            'applicable_weight': float(applicable), 'gross_error': bool(flags), 'flags': flags,
            'criteria': rows, 'sections': {name: {'total': rounded(v['earned']*100/v['max'])
                                                  if v['max'] else None,
                                                 'applicable_weight': float(v['max'])}
                                         for name, v in sections.items()}}
