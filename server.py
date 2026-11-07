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
from cqa.storage import Store, now


class JobCancelled(Exception):
    pass


ACTIVE_STATUSES = ('queued', 'running', 'cancelling')


def public_job(job):
    if not job: return None
    # Frozen transcripts are private storage, not polling payloads.
    result = {key: value for key, value in job.items() if key not in ['conversations', 'checklist']}
    result['retryable'] = job['total'] - len(job['results']) if job['status'] not in ACTIVE_STATUSES and not job.get('retry_job_id') else 0
    return result


class App:
    def __init__(self, store=None):
        self.store = store or Store()
        self.store.interrupt_jobs()
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        dev = ROOT / 'data/dev.json'
        if not self.store.conversations() and dev.exists():
            self.store.import_records(json.loads(dev.read_text()))

    def update(self, jid, **values):
        with self.lock:
            job = self.store.job(jid)
            job.update(values, updated=now())
            self.store.save_job(job)
            return job

    def job_state(self):
        with self.lock:
            jobs = self.store.recent_jobs()
            active = next((j for j in jobs if j['status'] in ACTIVE_STATUSES), None)
            return {'active_job': public_job(active), 'latest_job': public_job(jobs[0] if jobs else None)}

    def progress(self, jid, message):
        with self.lock:
            if self.store.job(jid)['status'] == 'cancelling': raise JobCancelled()
            self.update(jid, message=message)

    def start(self, ids, checklist_id, review=None):
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 100 or
                any(not isinstance(cid, str) or not cid.strip() for cid in ids) or len(set(ids)) != len(ids)):
            raise ValueError('Выберите 1–100 уникальных ID диалогов.')
        if review is not None and type(review) is not bool:
            raise ValueError('review должен быть логическим значением.')
        provider = Provider()
        checklist = self.store.checklist(checklist_id)
        conversations = [self.store.conversation(cid) for cid in ids]
        return self._submit(conversations, checklist, provider, provider.cfg['review'] if review is None else review)

    def retry(self, jid):
        with self.lock:
            original = self.store.job(jid)
            if original['status'] in ACTIVE_STATUSES: raise ValueError('Дождитесь завершения или отмены задачи.')
            done = {r['conversation_id'] for r in original['results']}
            pending = [c for c in original['conversations'] if c['conversation_id'] not in done]
            if not pending: raise ValueError('Все диалоги этой задачи уже оценены.')
            if original.get('retry_job_id'):
                raise ValueError('Повтор этой задачи уже создан. Продолжайте последнюю задачу.')
            result = self._submit(pending, original['checklist'], Provider(), original['review'], jid)
            self.update(jid, retry_job_id=result['job_id'])
            return result

    def cancel(self, jid):
        with self.lock:
            job = self.store.job(jid)
            if job['status'] in ACTIVE_STATUSES:
                job = self.update(jid, status='cancelling',
                    message='Отмена запрошена. Ожидаем завершения текущего запроса AI.')
            return public_job(job)

    def _submit(self, conversations, checklist, provider, review, parent_job_id=None):
        jid = str(uuid.uuid4())
        with self.lock:
            if self.job_state()['active_job']:
                raise ValueError('Дождитесь завершения текущей оценки.')
            stamp = now()
            job = {'id': jid, 'status': 'queued', 'total': len(conversations), 'completed': 0,
                   'results': [], 'errors': [], 'message': 'В очереди', 'created': stamp, 'updated': stamp,
                   'parent_job_id': parent_job_id, 'review': review,
                   'model': provider.cfg.get('model', ''),
                   'conversations': copy.deepcopy(conversations), 'checklist': copy.deepcopy(checklist)}
            self.store.save_job(job)
        self.pool.submit(self._work, jid, provider)
        return {'job_id': jid}

    def _work(self, jid, provider):
        try:
            self.progress(jid, 'Начинаем оценку')
            with self.lock:
                job = self.store.job(jid)
                if job['status'] == 'cancelling': raise JobCancelled()
                self.update(jid, status='running')
            for conversation in job['conversations']:
                cid = conversation['conversation_id']
                self.progress(jid, f'{cid}: подготовка')
                try:
                    result = evaluate(conversation, job['checklist'], provider, job['review'],
                                      lambda msg: self.progress(jid, f'{cid}: {msg}'))
                    # Store assessment and success checkpoint in one transaction: retry never
                    # duplicates a result saved just before a process crash.
                    with self.lock:
                        self.store.save_evaluation(conversation, job['checklist'], result, job_id=jid)
                except JobCancelled:
                    raise
                except Exception as err:
                    error = str(err) if isinstance(err, (ProviderError, ValueError)) else 'Внутренняя ошибка оценки.'
                    with self.lock:
                        current = self.store.job(jid)
                        errors = current['errors'] + [{'conversation_id': cid, 'error': error}]
                        self.update(jid, errors=errors, completed=len(current['results'])+len(errors))
            with self.lock:
                current = self.store.job(jid)
                status = 'completed_with_errors' if current['errors'] else 'completed'
                self.update(jid, status=status, message='Обработка завершена')
        except JobCancelled:
            self.update(jid, status='cancelled', message='Задача отменена. Сохранённые результаты доступны.')
        except Exception:
            self.update(jid, status='interrupted', message='Задача прервана. Можно повторить незавершённые диалоги.')


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
                        'checklists': app.store.checklists(), 'ai': public_config(), **app.job_state()})
                if url.path == '/api/conversation':
                    return self.send({'conversation': app.store.conversation(param('id')),
                                      'history': app.store.history(param('id'))})
                if url.path == '/api/evaluation': return self.send(app.store.evaluation(param('id')))
                if url.path == '/api/appeals': return self.send(app.store.appeals())
                if url.path == '/api/job':
                    with app.lock: job = app.store.job(param('id'))
                    return self.send(public_job(job))
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
                if path == '/api/evaluate':
                    if 'review' in body and type(body['review']) is not bool:
                        raise ValueError('review должен быть логическим значением.')
                    return self.send(app.start(body.get('ids'), body.get('checklist_id'), body.get('review')), 202)
                if path == '/api/job/cancel': return self.send(app.cancel(body.get('job_id')))
                if path == '/api/job/retry': return self.send(app.retry(body.get('job_id')), 202)
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
    port = int(os.getenv('PORT', '8765'))
    # Bind first: a second start on the same port must not interrupt live jobs.
    server = ThreadingHTTPServer(('127.0.0.1', port), BaseHTTPRequestHandler)
    app = App()
    server.RequestHandlerClass = handler_for(app)
    print(f'CQA запущен: http://127.0.0.1:{port}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        app.pool.shutdown(wait=False)
