import copy
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

from cqa.evaluator import evaluate
from cqa.provider import ProviderError
from server import App
from test_core import CHECKLIST, assessment, conversation
from test_http import HttpTests


class FakeProvider:
    cfg = {'review': True, 'model': 'test', 'provider': 'unit-test'}
    def complete(self, system, user, schema):
        return assessment(), {'input_tokens': 1, 'output_tokens': 1, 'seconds': 0, 'cost': 0}


class JobTests(HttpTests):
    def wait_done(self, jid):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            job = self.call('/api/job?id='+jid)
            if job['status'] not in ['queued', 'running', 'cancelling']:
                return job
            time.sleep(.01)
        self.fail('job did not finish')

    def test_invalid_checklist_is_400_without_new_version(self):
        original_count = len(self.store.checklists())
        bad = []
        for field, value in [('name', []), ('name', 'x'*201), ('criteria', [None]), ('criteria', ['text'])]:
            c = copy.deepcopy(CHECKLIST); c[field] = value; bad.append(c)
        for field, value in [('weight', True), ('weight', float('inf')), ('rule', []), ('na_rule', []), ('allow_na', 'true')]:
            c = copy.deepcopy(CHECKLIST); c['criteria'][0][field] = value; bad.append(c)
        for levels in [[None, None], [{'value': 0, 'description': []}, {'value': 10, 'description': 'ok'}]]:
            c = copy.deepcopy(CHECKLIST); c['criteria'][3]['levels'] = levels; bad.append(c)
        for body in bad:
            with self.subTest(body=body), self.assertRaises(urllib.error.HTTPError) as err:
                self.call('/api/checklists', body)
            self.assertEqual(err.exception.code, 400); err.exception.close()
        self.assertEqual(len(self.store.checklists()), original_count)

    def test_active_job_recovers_on_state_and_cancels_between_calls(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        class SlowProvider(FakeProvider):
            def complete(self, *args):
                calls.append(1); entered.set()
                release.wait(3)
                return super().complete(*args)
        try:
            with patch('server.Provider', SlowProvider):
                jid = self.call('/api/evaluate', {'ids': ['test']})['job_id']
                self.assertTrue(entered.wait(2))
                job = self.call('/api/state')['active_job']
                self.assertEqual(job['id'], jid)
                self.assertNotIn('conversations', job); self.assertNotIn('checklist', job)
                with self.assertRaises(urllib.error.HTTPError) as err:
                    self.call('/api/evaluate', {'ids': ['test']})
                self.assertEqual(err.exception.code, 400); err.exception.close()
                self.assertEqual(self.call('/api/job/cancel', {'job_id': jid})['status'], 'cancelling')
                release.set()
                result = self.wait_done(jid)
            self.assertEqual(result['status'], 'cancelled')
            self.assertEqual(result['retryable'], 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.store.history('test'), [])
        finally:
            release.set()

    def test_explicit_one_pass_and_invalid_review(self):
        for invalid in ['false', 1, None]:
            with self.assertRaises(urllib.error.HTTPError) as err:
                self.call('/api/evaluate', {'ids': ['test'], 'review': invalid})
            self.assertEqual(err.exception.code, 400); err.exception.close()
        with patch('server.Provider', FakeProvider):
            jid = self.call('/api/evaluate', {'ids': ['test'], 'review': False})['job_id']
            job = self.wait_done(jid)
        saved = self.store.evaluation(job['results'][0]['evaluation_id'])
        self.assertFalse(saved['reviewed']); self.assertEqual(saved['usage']['calls'], 1)
        self.assertFalse(job['review'])

    def test_retry_skips_success_and_preserves_frozen_inputs(self):
        second = conversation(); second['conversation_id'] = 'second'
        self.store.import_records([second])
        seen = []
        def fake_evaluate(conv, checklist, provider, review, progress):
            seen.append((conv['conversation_id'], copy.deepcopy(conv), copy.deepcopy(checklist), review))
            if conv['conversation_id'] == 'second' and len(seen) == 2:
                raise ProviderError('fixture timeout')
            return evaluate(conv, checklist, FakeProvider(), review=False)
        with patch('server.Provider', FakeProvider), patch('server.evaluate', fake_evaluate):
            jid = self.call('/api/evaluate', {'ids': ['test', 'second'], 'review': False})['job_id']
            first = self.wait_done(jid)
            self.assertEqual(first['status'], 'completed_with_errors')
            second['transcript'][0]['text'] = 'Changed after failure'
            self.store.import_records([second])
            changed = copy.deepcopy(CHECKLIST); changed['criteria'][0]['rule'] = 'Changed rule'
            self.store.save_checklist(changed)
            retry = self.call('/api/job/retry', {'job_id': jid})['job_id']
            done = self.wait_done(retry)
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(done['total'], 1)
        self.assertEqual([x[0] for x in seen], ['test', 'second', 'second'])
        self.assertEqual(seen[1][1:], seen[2][1:])
        self.assertEqual(len(self.store.history('test')), 1)
        self.assertEqual(len(self.store.history('second')), 1)
        self.assertEqual(self.call('/api/state')['latest_job']['id'], retry)
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.call('/api/job/retry', {'job_id': jid})
        self.assertEqual(err.exception.code, 400); err.exception.close()

    def test_restart_marks_unfinished_and_keeps_success_checkpoint(self):
        with patch('server.Provider', FakeProvider):
            jid = self.call('/api/evaluate', {'ids': ['test'], 'review': False})['job_id']
            completed = self.wait_done(jid)
        job = self.store.job(jid)
        second = conversation(); second['conversation_id'] = 'second'
        job['conversations'].append(second); job['total'] = 2; job['status'] = 'running'
        self.store.save_job(job)
        restarted = App(self.store)
        try:
            state = restarted.job_state()
            self.assertIsNone(state['active_job'])
            self.assertEqual(state['latest_job']['status'], 'interrupted')
            self.assertEqual(state['latest_job']['retryable'], 1)
            self.assertEqual(state['latest_job']['results'], completed['results'])
        finally:
            restarted.pool.shutdown(wait=True)

    def test_result_and_job_checkpoint_are_atomic(self):
        conv = conversation()
        result = evaluate(conv, CHECKLIST, FakeProvider(), review=False)
        with self.assertRaises(ValueError):
            self.store.save_evaluation(conv, CHECKLIST, result, job_id='missing-job')
        self.assertEqual(self.store.history('test'), [])

    def test_summary_flags_review_queue_and_checklist(self):
        conv = conversation(); checklist = self.store.checklist()
        result = evaluate(conv, checklist, FakeProvider(), review=False)
        result['assessment']['uncertainties'] = ['fixture uncertainty']
        saved = self.store.save_evaluation(conv, checklist, result)
        self.store.appeal(saved['id'], 'contact', 'fixture review')
        summary = self.call('/api/state')['conversations'][0]
        self.assertEqual(summary['checklist_id'], checklist['id'])
        self.assertTrue(summary['review_needed'])
        self.assertIn('Открытая апелляция', summary['review_reasons'])
        self.assertIn('Неопределённость AI', summary['review_reasons'])


def load_tests(loader, tests, pattern):
    # Reuse only the HTTP harness, not all inherited tests a second time.
    return unittest.TestSuite(JobTests(name) for name in JobTests.__dict__ if name.startswith('test_'))
