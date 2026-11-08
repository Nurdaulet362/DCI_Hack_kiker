"""Conservative review routing. Signals never assign or overwrite rubric answers."""
import re
import json
from pathlib import Path

DEFAULT_RULES = json.loads((Path(__file__).resolve().parent.parent / 'config/default_checklist.json').read_text())['criteria']
from .quality import contract_for

ROUTING_VERSION = 'adaptive-1-critical-and-contradictions'
RISK_TERMS = re.compile(
    r'верификац|идентификац|ограничени.{0,25}(?:сч[её]т|вывод)|'
    r'личн.{0,20}(?:карт|номер|плат|консультац)|читать умеете|'
    r'verif(?:y|ied|ication)|account.{0,25}(?:restrict|block)|private.{0,25}(?:paid|advice)|'
    r'жеке.{0,25}(?:ақылы|кеңес|нөмір)|сәйкестендір', re.I)
PUSHBACK = re.compile(
    r'я (?:уже|на компьютере)|какое приложение|не (?:этот|тот) раздел|'
    r'settings where|already (?:said|told)|which app|not (?:that|the) (?:section|app)|'
    r'қай (?:баптау|бөлім)|айттым|компьютер', re.I)
INTRODUCTION = re.compile(r'\b(?:я|меня зовут|менің атым|мен|my name|i am|i.m|this is)\b', re.I)


def review_reasons(assessment, checklist, conversation):
    reasons = []
    if assessment.get('uncertainties'): reasons.append('AI указал неопределённость')
    if conversation.get('channel') == 'voice call': reasons.append('Голосовой транскрипт')
    messages = conversation['transcript']
    text = '\n'.join(m['text'] for m in messages)
    if RISK_TERMS.search(text): reasons.append('Признаки идентификации или критического нарушения')
    clients = '\n'.join(m['text'] for m in messages if m['speaker'] == 'client')
    if PUSHBACK.search(clients): reasons.append('Возможный переспрос или исправление предположения оператора')
    if len(text) > 14000: reasons.append('Длинный транскрипт')
    for rule in checklist['criteria']:
        fields = ('kind', 'rule', 'na_rule', 'allow_na', 'levels')
        if not any(all(rule.get(k) == known.get(k) for k in fields) for known in DEFAULT_RULES):
            reasons.append('Новый или изменённый критерий')
        row = assessment['criteria'][rule['id']]
        checks = row.get('checks', {})
        if row['confidence'] < .85 or any(p['state'] == 'uncertain' for p in checks.values()):
            reasons.append('Низкая уверенность или неясный факт')
        if rule['kind'] == 'flag' and row['answer'] == 'yes': reasons.append('Флаг грубого нарушения')
        if rule['kind'] == 'binary' and row['answer'] == 'no': reasons.append('Невыполненный критерий')
        if rule['kind'] == 'scale' and row['answer'] != 'na' and float(row['answer']) < max(l['value'] for l in rule['levels']):
            reasons.append('Частичная или нулевая оценка')
        contract = contract_for(rule)
        if contract and contract['aggregation'] == 'identity':
            if checks.get('protected_action', {}).get('state') in ['present', 'uncertain']:
                reasons.append('Защищённое действие требует перепроверки')
        if contract and contract['aggregation'] == 'all' and 'introduction' in checks:
            proof = checks['introduction']
            quote = ' '.join(e['quote'] for e in proof['evidence'])
            if proof['state'] == 'present' and not INTRODUCTION.search(quote):
                reasons.append('Представление требует проверки смысла цитаты')
    return list(dict.fromkeys(reasons))
