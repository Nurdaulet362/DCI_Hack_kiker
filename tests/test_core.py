import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cqa.domain import calculate, normalize_conversation, validate_assessment, validate_checklist, digest
from cqa.evaluator import evaluate, response_schema
from cqa.settings import ROOT
from cqa.storage import Store


CHECKLIST = json.loads((ROOT / 'config/default_checklist.json').read_text())


def conversation():
    return normalize_conversation({'conversation_id': 'test', 'language': 'RU', 'channel': 'agent chat',
        'transcript': [{'speaker': 'operator', 'timestamp': '00:01', 'text': 'Здравствуйте! Я Анна.'},
                       {'speaker': 'client', 'timestamp': '00:04', 'text': 'Спасибо!'}]})


def assessment(checklist=CHECKLIST):
    result = {'criteria': {}, 'summary': 'Test fixture, not model output', 'uncertainties': []}
    for c in checklist['criteria']:
        result['criteria'][c['id']] = {'answer': str(max(l['value'] for l in c['levels'])) if c['kind'] == 'scale' else 'yes',
            'reason': 'Синтетическая заглушка только для проверки сервера.', 'confidence': 0.9,
            'support_type': 'observed', 'checked_start': 'm0001', 'checked_end': 'm0002',
            'evidence': [{'message_id': 'm0001', 'quote': 'Здравствуйте!'}]}
    result['criteria']['gross_error']['answer'] = 'no'
    return result


class CoreTests(unittest.TestCase):
    def test_full_score_and_flag_separate(self):
        a = assessment()
        a['criteria']['gross_error']['answer'] = 'yes'
        score = calculate(CHECKLIST, a)
        self.assertEqual(score['total'], 100)
        self.assertTrue(score['gross_error'])

    def test_scaled_weight(self):
        a = assessment()
        a['criteria']['management']['answer'] = '7'
        self.assertEqual(calculate(CHECKLIST, a)['total'], 95.5)

    def test_na_redistributes_and_rounds_once(self):
        a = assessment()
        a['criteria']['identity']['answer'] = 'na'
        a['criteria']['contact']['answer'] = 'no'
        self.assertEqual(calculate(CHECKLIST, a)['total'], 87.5)
        a['criteria']['needs']['answer'] = 'na'
        self.assertEqual(calculate(CHECKLIST, a)['total'], 84.62)

    def test_no_weight_means_no_total(self):
        c = {'name': 'Only flag', 'criteria': [CHECKLIST['criteria'][-1]]}
        self.assertIsNone(calculate(c, assessment())['total'])

    def test_discrete_levels_and_na_guard(self):
        for cid, answer in [('management', '8'), ('contact', 'na')]:
            a = assessment(); a['criteria'][cid]['answer'] = answer
            with self.assertRaises(ValueError): validate_assessment(a, CHECKLIST, conversation())

    def test_citations_must_exist_and_match(self):
        for field, value in [('message_id', 'm9999'), ('quote', 'Выдуманная цитата')]:
            a = assessment(); a['criteria']['contact']['evidence'][0][field] = value
            with self.assertRaises(ValueError): validate_assessment(a, CHECKLIST, conversation())

    def test_range_and_na_reason_type(self):
        a = assessment(); a['criteria']['needs']['answer'] = 'na'
        with self.assertRaises(ValueError): validate_assessment(a, CHECKLIST, conversation())
        a['criteria']['needs']['support_type'] = 'not_applicable'
        validate_assessment(a, CHECKLIST, conversation())
        a['criteria']['contact']['checked_start'] = 'm0002'
        with self.assertRaises(ValueError): validate_assessment(a, CHECKLIST, conversation())

    def test_checklist_validation_and_new_criterion_schema(self):
        c = copy.deepcopy(CHECKLIST)
        c['criteria'][0]['id'] = 'new_custom_id'
        c['criteria'][0]['title'] = 'Новый критерий'
        validate_checklist(c)
        schema = response_schema(c)
        self.assertIn('new_custom_id', schema['properties']['criteria']['properties'])
        c['criteria'][0]['weight'] = float('nan')
        with self.assertRaises(ValueError): validate_checklist(c)

    def test_atomic_import_and_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / 'test.db')
            conv = conversation()
            store.import_records([conv])
            with self.assertRaises(ValueError): store.import_records([conv, {'conversation_id': 'bad'}])
            self.assertEqual(len(store.conversations()), 1)
            result = {'assessment': assessment(), 'score': calculate(CHECKLIST, assessment()), 'transcript_hash': digest(conv)}
            saved = store.save_evaluation(conv, CHECKLIST, result)
            aid = store.appeal(saved['id'], 'contact', 'Оспариваю приветствие')
            with self.assertRaises(ValueError): store.appeal(saved['id'], 'contact', 'Дубликат')
            final = store.resolve(aid, 'no', 'Нет представления в первом ответе')
            self.assertEqual(final['score']['total'], 100)
            self.assertEqual(final['final_score']['total'], 90)
            self.assertEqual(final['assessment']['criteria']['contact']['answer'], 'yes')
            with self.assertRaises(ValueError): store.resolve(aid, 'yes', 'Повторное решение')
            conv['transcript'][0]['text'] = 'Другая реплика'
            store.import_records([conv])
            self.assertTrue(store.conversations()[0]['stale'])
            self.assertEqual(store.evaluation(saved['id'])['conversation']['transcript'][0]['text'], 'Здравствуйте! Я Анна.')

    def test_audit_and_repair_are_real_calls(self):
        draft = assessment()
        invalid = copy.deepcopy(draft)
        invalid['criteria']['contact']['evidence'][0]['quote'] = 'Invented'
        revised = copy.deepcopy(draft)
        revised['criteria']['contact']['answer'] = 'no'
        class FakeProvider:
            cfg = {'model': 'fixture', 'provider': 'unit-test'}
            def __init__(self): self.values = iter([invalid, draft, revised])
            def complete(self, system, user, schema):
                return next(self.values), {'input_tokens': 10, 'output_tokens': 20, 'seconds': 1, 'cost': None}
        result = evaluate(conversation(), CHECKLIST, FakeProvider())
        self.assertEqual(result['usage']['calls'], 3)
        self.assertEqual(result['score']['total'], 90)
        self.assertEqual(result['review_changes'], ['contact'])

    def test_real_dataset_integrity(self):
        path = ROOT / 'data/dev.json'
        if not path.exists(): self.skipTest('DEV not downloaded')
        records = json.loads(path.read_text())['data']['records']
        self.assertEqual(len(records), 65)
        for r in records: normalize_conversation(r)
        self.assertEqual(len({r['conversation_id'] for r in records}), 65)


if __name__ == '__main__': unittest.main()
