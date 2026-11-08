"""Synthetic protocol regressions, NOT a measurement of model accuracy."""
import copy
import io
import json
import os
import unittest
from unittest.mock import patch

from cqa.domain import normalize_conversation, validate_assessment
from cqa.evaluator import evaluate, response_schema
from cqa.provider import Provider, ProviderError
from cqa.quality import contract_for, proof_instructions, validate_proofs
from manage import reusable_result
from test_core import CHECKLIST, assessment, conversation


def sample(cid, rows, states, answer):
    rule = copy.deepcopy(next(c for c in CHECKLIST['criteria'] if c['id'] == cid))
    checklist = {'name': 'Synthetic proof regression', 'criteria': [rule]}
    conv = normalize_conversation({'conversation_id': 'random-unseen-id', 'language': 'RU',
        'channel': 'agent chat', 'transcript': [{'speaker': role, 'text': text, 'timestamp': ''}
                                              for role, text in rows]})
    def cite(index):
        msg = conv['transcript'][index]
        return [{'message_id': msg['id'], 'quote': msg['text']}]
    checks = {k: {'state': state, 'reason': 'Синтетическая проверка связи факта и ответа.',
                  'evidence': cite(index)} for k, (state, index) in states.items()}
    value = {'criteria': {cid: {'answer': answer, 'reason': 'Синтетическое обоснование.',
        'confidence': .8, 'support_type': 'not_applicable' if answer == 'na' else 'observed',
        'checked_start': conv['transcript'][0]['id'], 'checked_end': conv['transcript'][-1]['id'],
        'evidence': cite(0), 'checks': checks}}, 'summary': 'Synthetic', 'uncertainties': []}
    return value, checklist, conv


def validate(value, checklist, conv):
    return validate_proofs(validate_assessment(value, checklist, conv), checklist, conv)


class QualityTests(unittest.TestCase):
    def test_bot_cannot_supply_operator_greeting(self):
        args = sample('contact', [('bot', 'Здравствуйте, я помощник.'), ('operator', 'Какая ошибка?')],
            {'greeting': ('present', 0), 'introduction': ('present', 0)}, 'yes')
        with self.assertRaisesRegex(ValueError, 'operator'): validate(*args)

    def test_first_operator_is_not_first_message(self):
        for text in ['Здравствуйте! Меня зовут Мария.', 'Hello, I am Eva.', 'Сәлеметсіз бе! Менің атым Айша.']:
            args = sample('contact', [('bot', 'Connecting'), ('client', 'Help'), ('operator', text)],
                {'greeting': ('present', 2), 'introduction': ('present', 2)}, 'yes')
            validate(*args)
            self.assertEqual(proof_instructions(args[1], args[2])['first_operator_message_id'], 'm0003')

    def test_later_introduction_does_not_repair_first_reply(self):
        args = sample('contact', [('operator', 'Привет.'), ('operator', 'Я Мария.')],
            {'greeting': ('present', 0), 'introduction': ('present', 1)}, 'yes')
        with self.assertRaisesRegex(ValueError, 'первый ответ'): validate(*args)

    def test_missing_introduction_cannot_score_yes(self):
        args = sample('contact', [('operator', 'Добрый день! Как помочь?')],
            {'greeting': ('present', 0), 'introduction': ('absent', 0)}, 'yes')
        with self.assertRaisesRegex(ValueError, 'противоречит'): validate(*args)

    def test_client_thanks_and_system_close_are_not_operator_farewell(self):
        for role in ['client', 'system', 'bot']:
            args = sample('closing', [('operator', 'Заявка отправлена.'), (role, 'Спасибо, до свидания.')],
                {'opportunity': ('present', 0), 'summary_or_help': ('present', 0), 'farewell': ('present', 1)}, 'yes')
            with self.assertRaisesRegex(ValueError, 'operator'): validate(*args)

    def test_goodbye_alone_cannot_complete_conjunction(self):
        args = sample('closing', [('operator', 'До свидания!')],
            {'opportunity': ('present', 0), 'summary_or_help': ('absent', 0), 'farewell': ('present', 0)}, 'yes')
        with self.assertRaisesRegex(ValueError, 'противоречит'): validate(*args)

    def test_disconnect_before_opportunity_allows_na(self):
        args = sample('closing', [('operator', 'Откройте раздел отчётов.'), ('system', 'Connection lost')],
            {'opportunity': ('absent', 1), 'summary_or_help': ('absent', 0), 'farewell': ('absent', 1)}, 'na')
        validate(*args)

    def test_operator_close_with_opportunity_is_not_na(self):
        args = sample('closing', [('operator', 'Я закрываю чат.'), ('system', 'Operator closed chat')],
            {'opportunity': ('present', 1), 'summary_or_help': ('absent', 0), 'farewell': ('absent', 0)}, 'na')
        with self.assertRaisesRegex(ValueError, 'противоречит'): validate(*args)

    def test_each_gross_violation_suffices_without_customer_consent(self):
        for key, text in [('insult', 'Вы ничего не соображаете.'),
                          ('private_paid_offer', 'Pay me personally for private investment advice.'),
                          ('private_paid_offer', 'Ақылы жеке кеңес беремін.'),
                          ('confidential_disclosure', 'Вот чужой секретный код: 123.')]:
            states = {k: ('present' if k == key else 'absent', 0)
                      for k in ['insult', 'private_paid_offer', 'confidential_disclosure']}
            args = sample('gross_error', [('operator', text), ('client', 'Нет, не надо.')], states, 'no')
            with self.assertRaisesRegex(ValueError, 'противоречит'): validate(*args)
            args[0]['criteria']['gross_error']['answer'] = 'yes'
            validate(*args)

    def test_private_payment_warning_and_official_service_can_be_absent(self):
        states = {k: ('absent', 0) for k in ['insult', 'private_paid_offer', 'confidential_disclosure']}
        for text in ['Не платите сотрудникам лично за частные консультации.',
                     'The company offers a paid plan; pay through the official app.']:
            validate(*sample('gross_error', [('operator', text)], states, 'no'))

    def test_client_quoted_insult_cannot_prove_operator_violation(self):
        states = {k: ('present' if k == 'insult' else 'absent', 0)
                  for k in ['insult', 'private_paid_offer', 'confidential_disclosure']}
        args = sample('gross_error', [('client', 'Он сказал мне: дурак.')], states, 'yes')
        with self.assertRaisesRegex(ValueError, 'operator'): validate(*args)

    def test_guess_before_clarification_fails_needs(self):
        args = sample('needs', [('operator', 'Переустановите приложение.'), ('client', 'Я в браузере.')],
                      {'enough_initial': ('absent', 1), 'clarified_before_answer': ('absent', 0)}, 'yes')
        with self.assertRaisesRegex(ValueError, 'противоречит'): validate(*args)

    def test_unknown_facts_require_visible_uncertainty(self):
        args = sample('contact', [('operator', '[неразборчиво]')],
                      {'greeting': ('uncertain', 0), 'introduction': ('uncertain', 0)}, 'no')
        with self.assertRaisesRegex(ValueError, 'uncertainties'): validate(*args)
        args[0]['uncertainties'] = ['ASR не позволяет проверить приветствие.']
        validate(*args)
        args[0]['criteria']['contact']['answer'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'всех обязательных'): validate(*args)

    def test_absence_cannot_be_claimed_from_short_slice(self):
        args = sample('contact', [('operator', 'Здравствуйте.'), ('client', 'Привет.')],
                      {'greeting': ('present', 0), 'introduction': ('absent', 0)}, 'no')
        args[0]['criteria']['contact']['checked_end'] = 'm0001'
        with self.assertRaisesRegex(ValueError, 'всего диалога'): validate(*args)

    def test_renamed_rule_keeps_contract_but_edited_rule_does_not(self):
        rule = copy.deepcopy(CHECKLIST['criteria'][0]); rule['id'] = 'arbitrary'; rule['weight'] = 40
        self.assertIsNotNone(contract_for(rule))
        rule['rule'] = 'Достаточно приветствия бота. Имя не требуется.'
        self.assertIsNone(contract_for(rule))
        schema = response_schema({'criteria': [rule]})
        self.assertNotIn('checks', schema['properties']['criteria']['properties']['arbitrary']['properties'])

    def test_schema_fields_do_not_alias(self):
        schema = response_schema(CHECKLIST)
        row = schema['properties']['criteria']['properties']['contact']['properties']
        row['checked_start']['enum'] = ['m0001']
        self.assertNotIn('enum', row['reason'])
        self.assertNotIn('enum', row['checked_end'])
        self.assertNotIn('enum', schema['properties']['summary'])

    def test_facts_precede_answer_in_generated_schema(self):
        schema = response_schema(CHECKLIST)
        for rule in CHECKLIST['criteria']:
            row = schema['properties']['criteria']['properties'][rule['id']]
            if contract_for(rule):
                self.assertEqual(next(iter(row['properties'])), 'checks')
                self.assertLess(row['required'].index('checks'), row['required'].index('answer'))

    def test_enough_initial_explicitly_maps_to_na_even_with_clarification(self):
        args = sample('needs', [('client', 'Где скачать отчёт за август?'),
                               ('operator', 'Откройте Отчёты и выберите август.')],
                      {'enough_initial': ('present', 0), 'clarified_before_answer': ('present', 1)}, 'na')
        protocol = proof_instructions(args[1], args[2])
        first_mapping = protocol['answer_from_checks']['needs'][0]
        self.assertIn('enough_initial=present', first_mapping['if'])
        self.assertEqual(first_mapping['answer'], 'na')
        validate(*args)
        args[0]['criteria']['needs']['answer'] = 'yes'
        args[0]['criteria']['needs']['support_type'] = 'observed'
        with self.assertRaisesRegex(ValueError, 'противоречит'): validate(*args)

    def test_aggregation_mapping_follows_renamed_rules_not_ids(self):
        checklist = copy.deepcopy(CHECKLIST)
        rule = next(c for c in checklist['criteria'] if c['id'] == 'needs')
        rule['id'] = 'custom_request'
        self.assertIn('custom_request', proof_instructions(checklist, conversation())['answer_from_checks'])
        rule['rule'] = 'Уточнения не требуются; всегда оцени полноту ответа.'
        self.assertNotIn('custom_request', proof_instructions(checklist, conversation())['answer_from_checks'])

    def test_initial_request_excludes_operator_elicited_details_but_allows_voice_hello(self):
        args = sample('needs', [('client', 'Не могу открыть кабинет.'),
                               ('operator', 'Вы в приложении или браузере?'),
                               ('client', 'В браузере.'),
                               ('operator', 'Откройте страницу входа.')],
                      {'enough_initial': ('absent', 0), 'clarified_before_answer': ('present', 1)}, 'yes')
        validate(*args)
        spec = proof_instructions(args[1], args[2])['contracts']['needs']['enough_initial']['description']
        self.assertIn('ДО уточняющих вопросов оператора', spec)
        self.assertIn('Не включай сведения, полученные в ответ на уточняющий вопрос оператора', spec)
        self.assertIn('Приветствие оператора до первого обращения клиента не мешает N/A', spec)
        voice = sample('needs', [('operator', 'Здравствуйте! Я Мария.'),
                                ('client', 'Где скачать отчёт за август в веб-версии?'),
                                ('operator', 'Откройте Отчёты и выберите август.')],
                       {'enough_initial': ('present', 1), 'clarified_before_answer': ('absent', 2)}, 'na')
        voice[2]['channel'] = 'voice call'
        validate(*voice)

    def test_blind_review_never_receives_draft_and_disagreement_is_visible(self):
        first = assessment(); first['summary'] = 'UNIQUE_DRAFT_MARKER'
        second = copy.deepcopy(first); second['summary'] = 'Second pass'
        second['criteria']['contact']['answer'] = 'no'
        second['criteria']['contact']['checks']['introduction']['state'] = 'absent'
        class Fake:
            cfg = {}
            def __init__(self): self.inputs = []; self.values = iter([first, second])
            def complete(self, system, user, schema):
                self.inputs.append(user)
                return next(self.values), {'input_tokens': 1, 'output_tokens': 1, 'seconds': 0, 'cost': None}
        p = Fake(); result = evaluate(conversation(), CHECKLIST, p)
        self.assertNotIn('UNIQUE_DRAFT_MARKER', p.inputs[1])
        self.assertNotIn('conversation_id', p.inputs[0])
        self.assertEqual(result['usage']['calls'], 2)
        self.assertEqual(result['review_changes'], ['contact'])
        self.assertTrue(result['assessment']['uncertainties'])
        self.assertTrue(reusable_result(result, conversation(), CHECKLIST, result['experiment_config']))
        del result['assessment']['criteria']['contact']['checks']
        self.assertFalse(reusable_result(result, conversation(), CHECKLIST, result['experiment_config']))

    def test_invalid_proof_repairs_once_then_fails_closed(self):
        bad = assessment(); bad['criteria']['contact']['checks']['introduction']['state'] = 'absent'
        class Fake:
            cfg = {}
            calls = 0
            def complete(self, *args):
                self.calls += 1
                return copy.deepcopy(bad), {'input_tokens': 1, 'output_tokens': 1, 'seconds': 0, 'cost': None}
        p = Fake()
        with self.assertRaises(ProviderError): evaluate(conversation(), CHECKLIST, p, review=False)
        self.assertEqual(p.calls, 2)

    def test_ollama_receives_readable_schema_as_well_as_decoder_format(self):
        with patch.dict(os.environ, {'AI_PROVIDER': 'ollama', 'AI_MODEL': 'fixture',
                'AI_BASE_URL': 'http://127.0.0.1:11434', 'AI_REASONING_EFFORT': ''}), \
                patch('cqa.provider.open_api', return_value=io.BytesIO(b'{"message":{"content":"{}"}}')) as send:
            schema = response_schema(CHECKLIST)
            Provider().complete('audit', 'dialogue', schema)
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body['format'], schema)
        self.assertIn('private_paid_offer', body['messages'][0]['content'])

