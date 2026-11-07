#!/usr/bin/env python3
"""Local CQA server. Python 3.10+, no third-party dependencies."""
import copy
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

from cqa.settings import ROOT
from cqa.datasets import fetch_dataset
from cqa.evaluator import evaluate
from cqa.provider import Provider, public_config, ProviderError
from cqa.storage import Store


class App:
    def __init__(self, store=None):
        self.store = store or Store()
        self.jobs = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        dev = ROOT / 'data/dev.json'
        if not self.store.conversations() and dev.exists():
            self.store.import_records(json.loads(dev.read_text()))

    def update(self, jid, **values):
        with self.lock: self.jobs[jid].update(values)

    def start(self, ids, checklist_id):
        if not isinstance(ids, list) or not ids or len(ids) > 100 or len(set(ids)) != len(ids):
            raise ValueError('Выберите 1–100 уникальных диалогов.')
        provider = Provider()  # fail before accepting a job when AI is unconfigured
        checklist = self.store.checklist(checklist_id)
        # Freeze the source before queueing: edits/imports cannot change an in-flight run.
        conversations = [self.store.conversation(cid) for cid in ids]
        jid = str(uuid.uuid4())
        with self.lock:
            if any(j['status'] in ['queued', 'running'] for j in self.jobs.values()):
                raise ValueError('Дождитесь завершения текущей оценки.')
            self.jobs[jid] = {'id': jid, 'status': 'queued', 'total': len(ids),
                              'completed': 0, 'results': [], 'errors': [], 'message': 'В очереди'}

        def work():
            self.update(jid, status='running')
            results, errors = [], []
            for i, conversation in enumerate(conversations):
                cid = conversation['conversation_id']
                try:
                    result = evaluate(conversation, checklist, provider, provider.cfg['review'],
                        lambda msg: self.update(jid, message=f'{cid}: {msg}'))
                    saved = self.store.save_evaluation(conversation, checklist, result)
                    results.append({'conversation_id': cid, 'evaluation_id': saved['id']})
                except (ProviderError, ValueError) as e:
                    errors.append({'conversation_id': cid, 'error': str(e)})
                except Exception:
                    errors.append({'conversation_id': cid, 'error': 'Внутренняя ошибка оценки; проверьте локальную конфигурацию.'})
                self.update(jid, completed=i+1, results=results[:], errors=errors[:])
            self.update(jid, status='completed' if not errors else 'completed_with_errors', message='Обработка завершена')
        self.pool.submit(work)
        return {'job_id': jid}


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, fmt, *args):
            # Query strings may contain private IDs. Keep default console output quiet.
            pass

        def send(self, body, status=200, mime='application/json; charset=utf-8'):
            raw = json.dumps(body, ensure_ascii=False).encode() if mime.startswith('application/json') else body
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src https: http:; object-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                if self.headers.get('Host', '').split(':')[0] not in ['127.0.0.1', 'localhost']:
                    return self.send({'error': 'Недопустимый Host.'}, 403)
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                param = lambda key: query.get(key, [''])[0]
                if url.path == '/api/state':
                    return self.send({'conversations': app.store.conversations(),
                        'checklists': app.store.checklists(), 'ai': public_config()})
                if url.path == '/api/conversation':
                    return self.send({'conversation': app.store.conversation(param('id')),
                                      'history': app.store.history(param('id'))})
                if url.path == '/api/evaluation': return self.send(app.store.evaluation(param('id')))
                if url.path == '/api/appeals': return self.send(app.store.appeals())
                if url.path == '/api/job':
                    with app.lock: job = copy.deepcopy(app.jobs.get(param('id')))
                    if not job: raise ValueError('Задача не найдена (возможно, сервер перезапущен).')
                    return self.send(job)
                if url.path == '/favicon.ico': return self.send(b'', 204, 'image/x-icon')
                assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                          '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if url.path not in assets: return self.send({'error': 'Не найдено'}, 404)
                name, mime = assets[url.path]
                return self.send((ROOT / 'static' / name).read_bytes(), mime=mime)
            except ValueError as e: self.send({'error': str(e)}, 400)
            except Exception: self.send({'error': 'Ошибка сервера.'}, 500)

        def do_POST(self):
            try:
                # A localhost service without login still rejects cross-site writes.
                host = self.headers.get('Host', '')
                if host.split(':')[0] not in ['127.0.0.1', 'localhost']:
                    return self.send({'error': 'Недопустимый Host.'}, 403)
                origin = self.headers.get('Origin')
                if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                    return self.send({'error': 'Запрос с другого сайта запрещён.'}, 403)
                if origin and origin not in [f'http://{host}', f'https://{host}']:
                    return self.send({'error': 'Недопустимый Origin.'}, 403)
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    return self.send({'error': 'Используйте application/json.'}, 415)
                if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
                    return self.send({'error': 'Требуется один Content-Length без Transfer-Encoding.'}, 400)
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 20*1024*1024: raise ValueError('Максимальный размер запроса: 20 МБ.')
                body = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == '/api/import': return self.send({'imported': app.store.import_records(body)})
                if not isinstance(body, dict): raise ValueError('Ожидался JSON-объект.')
                if path == '/api/fetch':
                    count = app.store.import_records(fetch_dataset(body.get('group_id', 'DEV')))
                    return self.send({'imported': count})
                if path == '/api/checklists': return self.send(app.store.save_checklist(body))
                if path == '/api/evaluate': return self.send(app.start(body.get('ids'), body.get('checklist_id')), 202)
                if path == '/api/appeal':
                    return self.send({'id': app.store.appeal(body.get('evaluation_id'), body.get('criterion_id'), body.get('comment'))})
                if path == '/api/resolve':
                    return self.send(app.store.resolve(body.get('appeal_id'), body.get('answer'), body.get('reason')))
                self.send({'error': 'Не найдено'}, 404)
            except (ValueError, TypeError, KeyError, ProviderError) as e:
                self.send({'error': str(e)}, 400)
            except Exception:
                self.send({'error': 'Внутренняя ошибка сервера.'}, 500)
    return Handler


if __name__ == '__main__':
    app = App()
    port = int(os.getenv('PORT', '8765'))
    server = ThreadingHTTPServer(('127.0.0.1', port), handler_for(app))
    print(f'CQA запущен: http://127.0.0.1:{port}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        app.pool.shutdown(wait=False)
