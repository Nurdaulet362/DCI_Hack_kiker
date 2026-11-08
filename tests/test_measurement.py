import copy
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from cqa.benchmark import compare
from cqa.domain import digest
from cqa.evaluator import SYSTEM, evaluate, experiment_config
from manage import label_template, reusable_result, write_private_json
from test_core import CHECKLIST, assessment, conversation


class MeasurementTests(unittest.TestCase):
    def test_failed_and_partial_records_have_explicit_coverage(self):
        truth = [{'conversation_id': c, 'language': 'RU', 'criteria': {'contact': 'yes'}, 'total': 100}
                 for c in ['absent', 'failed', 'partial', 'valid']]
        predictions = [{'conversation_id': 'failed', 'error': 'timeout', 'score': {'total': 100}},
                       {'conversation_id': 'partial', 'assessment': {}, 'score': {'total': 100}},
                       {'conversation_id': 'valid', 'assessment': {'criteria': {'contact': {'answer': 'yes'}}},
                        'score': {'total': 100}, 'usage': {'seconds': 5}}]
        r = compare(predictions, truth)
        self.assertEqual(r['missing_records'], ['absent'])
        self.assertEqual(r['failed_records'], ['failed'])
        self.assertEqual(r['invalid_records'], ['partial'])
        self.assertEqual(r['matched_records'], 1)
        self.assertEqual(r['total_coverage'], .25)
        self.assertEqual(r['criterion_exact_match'], .25)
        self.assertEqual(r['total_mae'], 0)
        self.assertEqual(r['total_pairs'], 1)

    def test_gross_error_misses_include_missing_predictions(self):
        truth = [{'conversation_id': c, 'criteria': {'gross_error': answer}, 'total': None}
                 for c, answer in [('a', 'yes'), ('b', 'yes'), ('c', 'no')]]
        predictions = [{'conversation_id': c, 'assessment': {'criteria': {'gross_error': {'answer': 'yes'}}}}
                       for c in ['a', 'c']]
        r = compare(predictions, truth)
        self.assertEqual(r['gross_error']['precision'], .5)
        self.assertEqual(r['gross_error']['recall'], .5)
        self.assertEqual(r['gross_error']['unavailable_predictions'], 1)

    def test_unlabelled_and_nan_ground_truth_rejected(self):
        for labels, total in [({'contact': None}, None), ({'contact': 'yes'}, float('nan'))]:
            with self.assertRaises(ValueError):
                compare([], [{'conversation_id': 'test', 'criteria': labels, 'total': total}])

    def test_single_ui_export_supported(self):
        r = compare({'conversation_id': 'test', 'assessment': {'criteria': {'contact': {'answer': 'yes'}}},
                     'score': {'total': 100}},
                    [{'conversation_id': 'test', 'criteria': {'contact': 'yes'}, 'total': 100}])
        self.assertEqual(r['record_coverage'], 1)

    def test_annotation_template_never_copies_ai_answers(self):
        conv = conversation()
        conv['assessment'] = {'criteria': {'contact': {'answer': 'yes'}}}
        records = label_template([conv], CHECKLIST)
        self.assertNotIn('assessment', records[0])
        self.assertTrue(all(v is None for v in records[0]['criteria'].values()))
        self.assertIsNone(records[0]['total'])
        self.assertEqual(records[0]['answer_options']['contact'], ['yes', 'no'])
        with self.assertRaises(ValueError): compare([], records)

    def test_experiment_and_resume_require_matching_all_inputs(self):
        class Fixture:
            cfg = {'provider': 'unit-test', 'model': 'fixture', 'reasoning_effort': 'low',
                   'key': 'not-for-output', 'base_url': 'https://private.test'}
            def complete(self, system, user, schema):
                return assessment(), {'input_tokens': 1, 'output_tokens': 2, 'seconds': 3, 'cost': None}
        p = Fixture(); conv = conversation()
        result = evaluate(conv, CHECKLIST, p, review=True)
        self.assertEqual(result['review_mode'], 'blind_same_model_review')
        self.assertEqual([c['stage'] for c in result['usage']['per_call']], ['initial', 'review'])
        self.assertNotIn('not-for-output', json.dumps(result))
        self.assertNotIn('private.test', json.dumps(result))
        self.assertEqual(result['prompt_hash'], digest(SYSTEM))
        exp = experiment_config(p, True)
        self.assertTrue(reusable_result(result, conv, CHECKLIST, exp))
        changed = copy.deepcopy(conv); changed['transcript'][0]['text'] += '!'
        self.assertFalse(reusable_result(result, changed, CHECKLIST, exp))
        changed_exp = {**exp, 'reasoning_effort': 'high'}
        self.assertFalse(reusable_result(result, conv, CHECKLIST, changed_exp))
        changed_checklist = copy.deepcopy(CHECKLIST); changed_checklist['criteria'][0]['rule'] += '!'
        self.assertFalse(reusable_result(result, conv, changed_checklist, exp))

    def test_checkpoint_atomic_and_private(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'results.json'
            write_private_json(path, [{'conversation_id': 'one'}])
            with self.assertRaises(ValueError): write_private_json(path, {'bad': float('nan')})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), [{'conversation_id': 'one'}])
            if os.name == 'posix': self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(len(list(Path(d).iterdir())), 1)

class CheckpointTrustTests(unittest.TestCase):
    def setUp(self):
        class Fixture:
            cfg = {'provider': 'fixture', 'model': 'fixture'}
            def complete(self, system, user, schema):
                return assessment(), {'input_tokens': 1, 'output_tokens': 2, 'seconds': 1, 'cost': None}
        self.conv = conversation()
        self.exp = experiment_config(Fixture(), False)
        self.result = evaluate(self.conv, CHECKLIST, Fixture(), review=False)

    def test_matching_hashes_do_not_admit_incomplete_or_forged_scores(self):
        self.assertTrue(reusable_result(self.result, self.conv, CHECKLIST, self.exp))
        variants = []
        empty = copy.deepcopy(self.result); empty['assessment'] = {'criteria': {}}; empty.pop('score'); variants.append(empty)
        missing_score = copy.deepcopy(self.result); missing_score.pop('score'); variants.append(missing_score)
        wrong_score = copy.deepcopy(self.result); wrong_score['score']['total'] = 0; variants.append(wrong_score)
        wrong_points = copy.deepcopy(self.result); wrong_points['score']['criteria'][0]['points'] = 0; variants.append(wrong_points)
        invented_quote = copy.deepcopy(self.result)
        invented_quote['assessment']['criteria']['contact']['evidence'][0]['quote'] = 'This was never said'
        variants.append(invented_quote)
        for record in variants:
            with self.subTest(assessment=record['assessment'], score=record.get('score')):
                self.assertFalse(reusable_result(record, self.conv, CHECKLIST, self.exp))

    def test_embedded_snapshots_must_match_even_if_declared_hashes_match(self):
        record = copy.deepcopy(self.result)
        record['conversation'] = copy.deepcopy(self.conv)
        record['conversation']['transcript'][0]['text'] = 'Outdated transcript'
        self.assertFalse(reusable_result(record, self.conv, CHECKLIST, self.exp))
        record['conversation'] = self.conv
        record['checklist'] = copy.deepcopy(CHECKLIST)
        record['checklist']['criteria'][0]['rule'] = 'A different rule'
        self.assertFalse(reusable_result(record, self.conv, CHECKLIST, self.exp))

    def test_benchmark_hash_conflicts_cannot_score_or_contribute_latency(self):
        for field in ['transcript_hash', 'checklist_hash']:
            prediction = {'conversation_id': 'test', field: 'old',
                'assessment': {'criteria': {'gross_error': {'answer': 'yes'}}},
                'score': {'total': 100}, 'usage': {'seconds': 1}}
            truth = {'conversation_id': 'test', field: 'new',
                     'criteria': {'gross_error': 'yes'}, 'total': 100}
            with self.subTest(field=field):
                result = compare([prediction], [truth])
                self.assertEqual(result['incompatible_records'], ['test'])
                self.assertEqual(result['incompatible_fields'], {'test': [field]})
                self.assertEqual(result['record_coverage'], 0)
                self.assertEqual(result['total_coverage'], 0)
                self.assertIsNone(result['total_mae'])
                self.assertEqual(result['criterion_exact_match'], 0)
                self.assertEqual(result['gross_error']['false_negative'], 1)
                self.assertEqual(result['latency_seconds']['samples'], 0)
                self.assertEqual(result['invalid_records'], [])

    def test_compatible_and_legacy_hashes_still_score(self):
        prediction = {'conversation_id': 'test', 'transcript_hash': 'current', 'checklist_hash': 'rules',
            'assessment': {'criteria': {'contact': {'answer': 'yes'}}}, 'score': {'total': 100}}
        for hashes in [{}, {'transcript_hash': 'current', 'checklist_hash': 'rules'}]:
            result = compare([prediction], [{'conversation_id': 'test', 'criteria': {'contact': 'yes'},
                                            'total': 100, **hashes}])
            self.assertEqual(result['record_coverage'], 1)
            self.assertEqual(result['total_mae'], 0)
            self.assertEqual(result['incompatible_records'], [])
