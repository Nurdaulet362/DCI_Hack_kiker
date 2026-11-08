"""Protocol and request-count tests only, not evidence of LLM accuracy."""
import copy
import tempfile
import unittest
from pathlib import Path
from cqa.evaluator import evaluate, experiment_config
from cqa.routing import review_reasons
from cqa.storage import Store
from test_core import CHECKLIST, conversation, assessment
from test_quality import sample, validate


def neutral():
    value = assessment()
    value['criteria']['contact']['checks']['introduction']['evidence'][0]['quote'] = 'Здравствуйте! Я Анна.'
    row = value['criteria']['identity']
    row['answer'] = 'na'; row['support_type'] = 'not_applicable'
    for proof in row['checks'].values(): proof['state'] = 'absent'
    return value


class Fixture:
    cfg = {'provider': 'fixture', 'model': 'synthetic'}
    def __init__(self, value): self.value = value; self.calls = 0
    def complete(self, *args):
        self.calls += 1
        return copy.deepcopy(self.value), {'input_tokens': 10, 'output_tokens': 10, 'seconds': 1, 'cost': None}


class AdaptiveTests(unittest.TestCase):
    def test_plain_case_one_call_and_actual_reviewed_false(self):
        p = Fixture(neutral()); result = evaluate(conversation(), CHECKLIST, p, review='auto')
        self.assertEqual(p.calls, 1)
        self.assertFalse(result['reviewed'])
        self.assertEqual(result['review_decision']['reasons'], [])
        self.assertEqual(result['review_mode'], 'adaptive_review')

    def test_voice_always_gets_two_calls(self):
        conv = conversation(); conv['channel'] = 'voice call'
        p = Fixture(neutral()); result = evaluate(conv, CHECKLIST, p, review='auto')
        self.assertEqual(p.calls, 2); self.assertTrue(result['reviewed'])

    def test_single_and_full_still_work(self):
        for mode, expected in [(False, 1), (True, 2)]:
            p = Fixture(neutral()); evaluate(conversation(), CHECKLIST, p, review=mode)
            self.assertEqual(p.calls, expected)

    def test_invalid_mode_rejected_before_paid_call(self):
        p = Fixture(neutral())
        with self.assertRaises(ValueError): evaluate(conversation(), CHECKLIST, p, review='anything')
        self.assertEqual(p.calls, 0)

    def test_customer_pushback_routes_without_changing_answers(self):
        a = neutral(); before = copy.deepcopy(a); conv = conversation()
        for text in ['Я на компьютере вообще.', 'Settings where?', 'Мен компьютер арқылы кірдім.']:
            conv['transcript'][1]['text'] = text
            self.assertTrue(review_reasons(a, CHECKLIST, conv))
        self.assertEqual(a, before)

    def test_new_rule_and_uncertainty_route_to_review(self):
        a = neutral(); rules = copy.deepcopy(CHECKLIST)
        rules['criteria'][0]['rule'] = 'Достаточно приветствия.'
        self.assertIn('Новый или изменённый критерий', review_reasons(a, rules, conversation()))
        a['uncertainties'] = ['Неясная расшифровка.']
        self.assertIn('AI указал неопределённость', review_reasons(a, CHECKLIST, conversation()))

    def test_greeting_question_is_routed_not_silently_relabelled(self):
        a = neutral(); a['criteria']['contact']['checks']['introduction']['evidence'][0]['quote'] = 'Сәлем. Қай сайтқа кірдіңіз?'
        self.assertIn('Представление требует проверки смысла цитаты', review_reasons(a, CHECKLIST, conversation()))
        self.assertEqual(a['criteria']['contact']['answer'], 'yes')

    def test_disclosure_without_verification_cannot_be_na_or_yes(self):
        for bad in ['yes', 'na']:
            args = sample('identity', [('operator', 'На вашем счёте действует ограничение вывода.'),
                ('client', 'Но я не подтверждал личность.')],
                {'protected_action': ('present', 0), 'verified_before_action': ('absent', 1)}, bad)
            with self.assertRaisesRegex(ValueError, 'противоречит'): validate(*args)
            args[0]['criteria']['identity']['answer'] = 'no'
            args[0]['criteria']['identity']['support_type'] = 'absence'
            validate(*args)

    def test_general_tariff_query_can_be_na(self):
        validate(*sample('identity', [('client', 'Где посмотреть общие тарифы?'), ('operator', 'На сайте.')],
            {'protected_action': ('absent', 1), 'verified_before_action': ('absent', 0)}, 'na'))

    def test_client_disclosure_cannot_prove_operator_action(self):
        args = sample('identity', [('client', 'На моём счёте ограничение.')],
            {'protected_action': ('present', 0), 'verified_before_action': ('absent', 0)}, 'no')
        with self.assertRaisesRegex(ValueError, 'operator'): validate(*args)

    def test_resume_modes_have_different_fingerprints(self):
        self.assertNotEqual(experiment_config(Fixture(neutral()), 'auto'), experiment_config(Fixture(neutral()), True))

    def test_null_import_is_validation_error(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d)/'test.sqlite3')
            with self.assertRaises(ValueError): store.import_records({'data': None})

    def test_late_verification_is_rejected(self):
        args = sample('identity', [('operator', 'На вашем счёте ограничение вывода.'),
                                  ('system', 'Identity verified')],
            {'protected_action': ('present', 0), 'verified_before_action': ('present', 1)}, 'yes')
        with self.assertRaisesRegex(ValueError, 'предшествовать'): validate(*args)
