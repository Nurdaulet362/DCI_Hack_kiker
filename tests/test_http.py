import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from cqa.benchmark import compare
from cqa.provider import Provider, ProviderError
from cqa.storage import Store
from server import App, handler_for
from test_core import assessment, conversation, CHECKLIST


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name)/'db.sqlite3')
        self.store.import_records([conversation()])
        self.app = App(self.store)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.app))
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.app.pool.shutdown(wait=True); self.tmp.cleanup()

    def call(self, path, body=None, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin: headers['Origin'] = origin
        request = urllib.request.Request(self.url+path, None if body is None else json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request) as r: return json.load(r)

    def test_missing_provider_cannot_generate_fake_score(self):
        with patch.dict(os.environ, {'AI_API_KEY': '', 'OPENAI_API_KEY': '', 'AI_MODEL': ''}):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.call('/api/evaluate', {'ids': ['test']})
            self.assertEqual(e.exception.code, 400)
            e.exception.close()
        self.assertFalse(self.call('/api/conversation?id=test')['history'])

    def test_cross_origin_write_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call('/api/import', [conversation()], origin='https://evil.example')
        self.assertEqual(e.exception.code, 403)
        e.exception.close()

    def test_end_to_end_job_appeal_export_and_checklist_version(self):
        class FakeProvider:
            cfg = {'review': True, 'model': 'test', 'provider': 'unit-test'}
            def complete(self, system, user, schema):
                return assessment(), {'input_tokens': 1, 'output_tokens': 1, 'seconds': 0, 'cost': 0}
        with patch('server.Provider', FakeProvider):
            job = self.call('/api/evaluate', {'ids': ['test']})['job_id']
            for _ in range(100):
                result = self.call('/api/job?id='+job)
                if result['status'].startswith('completed'): break
                time.sleep(.01)
        self.assertEqual(result['status'], 'completed', result)
        eid = result['results'][0]['evaluation_id']
        original = self.call('/api/evaluation?id='+eid)
        self.assertEqual(original['score']['total'], 100)
        aid = self.call('/api/appeal', {'evaluation_id': eid, 'criterion_id': 'management', 'comment': 'Проверьте связность'})['id']
        final = self.call('/api/resolve', {'appeal_id': aid, 'answer': '4', 'reason': 'Клиент переспрашивал'})
        self.assertEqual(final['final_score']['total'], 91)
        self.assertEqual(final['score']['total'], 100)
        self.assertEqual(self.call('/api/appeals')[0]['status'], 'resolved')
        new = json.loads(json.dumps(CHECKLIST)); new['criteria'][0]['rule'] = 'Новый критерий'
        self.call('/api/checklists', new)
        self.assertEqual(self.call('/api/evaluation?id='+eid)['checklist']['criteria'][0]['rule'], CHECKLIST['criteria'][0]['rule'])

    def test_benchmark_missing_coverage(self):
        result = compare([{'conversation_id':'a', 'assessment':{'criteria':{'id':{'answer':'yes'}}}, 'score':{'total':80}}],
                         [{'conversation_id':'a','criteria':{'id':'yes'},'total':90}, {'conversation_id':'b','criteria':{'id':'no'},'total':100}])
        self.assertEqual(result['criterion_exact_match'], .5)
        self.assertEqual(result['total_mae'], 10)
        self.assertEqual(result['missing_records'], ['b'])


if __name__ == '__main__': unittest.main()
