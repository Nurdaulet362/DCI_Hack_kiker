"""Versioned proof contracts for known rules; never infer a rule from its ID.

These checks validate attribution and logical consistency, not linguistic truth.
Unknown/edited rubrics retain generic evaluation instead of inheriting old rules.
"""
import json
from pathlib import Path

CONTRACT_VERSION = 'proof-contracts-2-explicit-aggregation'
CONTRACTS = json.loads((Path(__file__).with_name('proof_contracts.json')).read_text())

# Explain the same logic that validate_proofs enforces. These are rubric mappings,
# not labels for any specific transcript; changed rules receive no old mapping.
AGGREGATION_INSTRUCTIONS = {
    'all': [
        {'if': 'Все checks имеют state=present.', 'answer': 'yes'},
        {'if': 'Хотя бы один check имеет state=absent.', 'answer': 'no'},
        {'if': 'Нет absent, но есть uncertain.', 'answer': 'no',
         'note': 'Укажи конкретное сомнение в uncertainties; Да требует подтверждения всех компонентов.'}],
    'any': [
        {'if': 'Хотя бы один check имеет state=present.', 'answer': 'yes'},
        {'if': 'Все checks имеют state=absent.', 'answer': 'no'},
        {'if': 'Нет present, но есть uncertain.', 'answer': 'no',
         'note': 'Укажи сомнение в uncertainties. Не утверждай, что нарушение достоверно исключено.'}],
    'needs': [
        {'if': 'enough_initial=present (данных хватало сразу), независимо от clarified_before_answer.',
         'answer': 'na', 'note': 'Это неприменимость: уточнять ничего не требовалось. Нельзя выбрать yes.'},
        {'if': 'enough_initial=absent И clarified_before_answer=present.', 'answer': 'yes'},
        {'if': 'enough_initial=absent И clarified_before_answer=absent.', 'answer': 'no'},
        {'if': 'Иначе: применимость или нужное уточнение не установлены.', 'answer': 'no',
         'note': 'Укажи сомнение в uncertainties. Не утверждай подтверждённость выполнения.'}],
    'closing': [
        {'if': 'summary_or_help=present И farewell=present.', 'answer': 'yes'},
        {'if': 'Не выполнено предыдущее условие И opportunity=absent.', 'answer': 'na'},
        {'if': 'opportunity=present И хотя бы один из summary_or_help/farewell имеет state=absent.',
         'answer': 'no'},
        {'if': 'Иначе: для выбора yes или na не хватает подтверждений.', 'answer': 'no',
         'note': 'Укажи конкретную неопределённость в uncertainties.'}]
}


def contract_for(rule):
    fields = ('kind', 'rule', 'na_rule', 'allow_na', 'levels')
    return next((c for c in CONTRACTS if all(rule.get(k) == c['rule'].get(k)
                                            for k in fields)), None)


def proof_instructions(checklist, conversation):
    operators = [m['id'] for m in conversation['transcript'] if m['speaker'] == 'operator']
    return {
        'first_operator_message_id': operators[0] if operators else None,
        'last_operator_message_id': operators[-1] if operators else None,
        'contracts': {c['id']: contract['checks'] for c in checklist['criteria']
                      if (contract := contract_for(c))},
        'answer_from_checks': {c['id']: AGGREGATION_INSTRUCTIONS[contract['aggregation']]
                              for c in checklist['criteria'] if (contract := contract_for(c))},
        'protocol': (
            'Для каждого checks независимо установи present/absent/uncertain. '
            'Это наличие факта, не итоговая оценка. Сначала заполни checks с reason и evidence; '
            'только после этого выбери answer по answer_from_checks. Условия проверяй сверху вниз. '
            'Если answer=na, support_type обязательно not_applicable; для других ответов '
            'support_type observed или absence. Не выбирай итог до проверки фактов. '
            'Клиентская благодарность и системное закрытие не являются прощанием оператора. '
            'Первый ответ оператора определяется ролью, а не местом в общем диалоге. '
            'Если есть составное требование, докажи каждую часть отдельно. '
            'Проверяй каждую категорию нарушения во всех репликах оператора, включая RU/KZ/EN. '
            'Отказ клиента не отменяет уже сделанного запрещённого предложения. '
            'Оператор сам закрыл чат — это не доказательство внезапного обрыва. '
            'Неопределённость факта требует записи в uncertainties. '
            'Для отсутствия проверяй весь диалог: checked_start/end — первая/последняя реплика. '
            'Не подменяй проверку смысла поиском ключевых слов: отрицание, цитирование жалобы '
            'и запрет частной услуги не являются предложением этой услуги.')}


def validate_proofs(result, checklist, conversation):
    messages = {m['id']: m for m in conversation['transcript']}
    order = {mid: i for i, mid in enumerate(messages)}
    operators = [m['id'] for m in messages.values() if m['speaker'] == 'operator']
    for rule in checklist['criteria']:
        contract = contract_for(rule)
        if not contract:
            continue
        cid = rule['id']
        row = result['criteria'][cid]
        proofs = row.get('checks')
        if not isinstance(proofs, dict) or set(proofs) != set(contract['checks']):
            raise ValueError(f'{cid}: нужны отдельные checks для всех компонентов правила.')
        states = {}
        for key, spec in contract['checks'].items():
            proof = proofs[key]
            if not isinstance(proof, dict) or proof.get('state') not in ['present', 'absent', 'uncertain']:
                raise ValueError(f'{cid}/{key}: неверный state.')
            if not isinstance(proof.get('reason'), str) or not proof['reason'].strip():
                raise ValueError(f'{cid}/{key}: нужно содержательное обоснование.')
            evidence = proof.get('evidence')
            if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
                raise ValueError(f'{cid}/{key}: нужны 1–3 цитаты факта или контекста отсутствия.')
            for e in evidence:
                if not isinstance(e, dict):
                    raise ValueError(f'{cid}/{key}: неверная цитата.')
                mid, quote = e.get('message_id'), e.get('quote')
                if mid not in messages or not isinstance(quote, str) or not quote.strip() or quote not in messages[mid]['text']:
                    raise ValueError(f'{cid}/{key}: цитата не найдена.')
                if not order[row['checked_start']] <= order[mid] <= order[row['checked_end']]:
                    raise ValueError(f'{cid}/{key}: цитата вне проверенного участка.')
                if proof['state'] == 'present' and spec.get('speaker') and messages[mid]['speaker'] != spec['speaker']:
                    raise ValueError(f'{cid}/{key}: действие должен подтверждать сам {spec["speaker"]}, не клиент/бот/system.')
                if proof['state'] == 'present' and spec.get('first_operator') and (not operators or mid != operators[0]):
                    raise ValueError(f'{cid}/{key}: нужен первый ответ оператора, не более поздняя реплика.')
            if proof['state'] != 'present' and (row['checked_start'] != next(iter(messages)) or row['checked_end'] != next(reversed(messages))):
                raise ValueError(f'{cid}/{key}: отсутствие/неопределённость требует проверки всего диалога.')
            if proof['state'] == 'uncertain' and not any(u.strip() for u in result['uncertainties']):
                raise ValueError(f'{cid}/{key}: укажи сомнение в uncertainties.')
            states[key] = proof['state']
        expected = None
        kind = contract['aggregation']
        if kind == 'all':
            if all(v == 'present' for v in states.values()): expected = 'yes'
            elif 'absent' in states.values(): expected = 'no'
            elif row['answer'] == 'yes':
                raise ValueError(f'{cid}: Да требует подтверждения всех обязательных компонентов.')
        elif kind == 'any':
            if 'present' in states.values(): expected = 'yes'
            elif all(v == 'absent' for v in states.values()): expected = 'no'
            elif row['answer'] == 'yes':
                raise ValueError(f'{cid}: нарушение требует хотя бы одного подтверждённого факта.')
        elif kind == 'closing':
            if states['summary_or_help'] == states['farewell'] == 'present': expected = 'yes'
            elif states['opportunity'] == 'absent': expected = 'na'
            elif states['opportunity'] == 'present' and 'absent' in (states['summary_or_help'], states['farewell']): expected = 'no'
            elif row['answer'] == 'yes':
                raise ValueError(f'{cid}: Да требует и итога/помощи, и прощания оператора.')
            elif row['answer'] == 'na':
                raise ValueError(f'{cid}: N/A требует подтверждённого отсутствия возможности завершить чат.')
        elif kind == 'needs':
            if states['enough_initial'] == 'present': expected = 'na'
            elif states['enough_initial'] == 'absent':
                if states['clarified_before_answer'] == 'present': expected = 'yes'
                elif states['clarified_before_answer'] == 'absent': expected = 'no'
            if expected is None and row['answer'] in ['yes', 'na']:
                raise ValueError(f'{cid}: недостаточно подтверждений для Да или N/A.')
        if expected is not None and row['answer'] != expected:
            raise ValueError(f'{cid}: answer={row["answer"]} противоречит checks; по указанным фактам ответ {expected}. Перепроверь факты по диалогу, не меняй их ради сохранения ответа.')
    return result
