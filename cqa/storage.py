import copy
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .domain import calculate, digest, normalize_conversation, validate_checklist, choices
from .settings import ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path=None):
        self.path = str(path or ROOT / 'data/cqa.sqlite3')
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == 'posix': os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, body TEXT NOT NULL, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS checklists (id TEXT PRIMARY KEY, body TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                body TEXT NOT NULL, created TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS eval_conversation ON evaluations(conversation_id, created);
            CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS appeals (id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL,
                criterion_id TEXT NOT NULL, comment TEXT NOT NULL, status TEXT NOT NULL,
                answer TEXT, reason TEXT, created TEXT NOT NULL, resolved TEXT);
            ''')
            if not db.execute('SELECT id FROM checklists LIMIT 1').fetchone():
                initial = json.loads((ROOT / 'config/default_checklist.json').read_text(encoding='utf-8'))
                self._save_checklist(db, initial)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _save_checklist(self, db, body):
        validate_checklist(body)
        cid = str(uuid.uuid4())
        db.execute('INSERT INTO checklists VALUES (?,?,?)', (cid, json.dumps(body, ensure_ascii=False), now()))
        return {'id': cid, **body}

    def save_checklist(self, body):
        validate_checklist(body)
        body = {k: v for k, v in body.items() if k not in ['id', 'created']}
        with self.db() as db:
            return self._save_checklist(db, body)

    def checklists(self):
        with self.db() as db:
            return [{'id': r['id'], 'created': r['created'], **json.loads(r['body'])}
                    for r in db.execute('SELECT * FROM checklists ORDER BY rowid DESC')]

    def checklist(self, cid=None):
        lists = self.checklists()
        if not cid: return lists[0]
        for c in lists:
            if c['id'] == cid: return c
        raise ValueError('Чек-лист не найден.')

    def import_records(self, payload):
        if isinstance(payload, list): records = payload
        elif isinstance(payload, dict):
            if payload.get('isSuccess') is False:
                raise ValueError('API организаторов сообщил об ошибке.')
            data = payload.get('data')
            if 'data' in payload and not isinstance(data, dict):
                raise ValueError('data должен быть объектом с массивом records.')
            records = data.get('records') if 'data' in payload else payload.get('records')
            if records is None and 'conversation_id' in payload: records = [payload]
        else: records = None
        if not isinstance(records, list) or not 1 <= len(records) <= 1000:
            raise ValueError('Импортируйте массив из 1–1000 диалогов или ответ API с data.records.')
        normalized = [normalize_conversation(r) for r in records]
        ids = [r['conversation_id'] for r in normalized]
        if len(set(ids)) != len(ids):
            raise ValueError('В наборе есть повторяющиеся conversation_id.')
        with self.db() as db:
            for r in normalized:
                db.execute('INSERT INTO conversations VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body,hash=excluded.hash',
                           (r['conversation_id'], json.dumps(r, ensure_ascii=False), digest(r)))
        return len(normalized)

    def conversation(self, cid):
        with self.db() as db:
            row = db.execute('SELECT body FROM conversations WHERE id=?', (cid,)).fetchone()
            if not row: raise ValueError('Диалог не найден.')
            return json.loads(row['body'])

    def conversations(self):
        with self.db() as db:
            records = [json.loads(r['body']) for r in db.execute('SELECT body FROM conversations ORDER BY id')]
            summaries = []
            for r in records:
                last = db.execute('SELECT id, body FROM evaluations WHERE conversation_id=? ORDER BY rowid DESC LIMIT 1',
                                  (r['conversation_id'],)).fetchone()
                result = self.evaluation(last['id']) if last else None
                reasons = []
                if result:
                    if result['final_score']['gross_error']: reasons.append('Грубое нарушение')
                    if result['final_assessment'].get('uncertainties'): reasons.append('Неопределённость AI')
                    if any(r.get('confidence', 1) < .75 for r in result['final_assessment']['criteria'].values()):
                        reasons.append('Низкая уверенность AI')
                    if result.get('review_changes'): reasons.append('Расхождение двух проходов')
                    if any(a['status'] == 'open' for a in result['appeals']): reasons.append('Открытая апелляция')
                summaries.append({k: v for k, v in r.items() if k != 'transcript'} | {
                    'messages': len(r['transcript']),
                    'preview': next((m['text'] for m in r['transcript'] if m['speaker'] == 'client'), ''),
                    'evaluation_id': last['id'] if last else None,
                    'checklist_id': result['checklist'].get('id') if result else None,
                    'review_needed': bool(reasons), 'review_reasons': reasons,
                    'score': result['final_score'] if result else None,
                    'stale': result['transcript_hash'] != digest(r) if result else False})
            return summaries

    def save_evaluation(self, conversation, checklist, result, job_id=None):
        eid = str(uuid.uuid4())
        body = {**result, 'id': eid, 'conversation_id': conversation['conversation_id'],
                'conversation': conversation, 'checklist': checklist, 'created': now()}
        with self.db() as db:
            db.execute('INSERT INTO evaluations VALUES (?,?,?,?)',
                       (eid, conversation['conversation_id'], json.dumps(body, ensure_ascii=False), body['created']))
            if job_id:
                row = db.execute('SELECT body FROM jobs WHERE id=?', (job_id,)).fetchone()
                if not row: raise ValueError('Задача не найдена.')
                job = json.loads(row['body'])
                job['results'].append({'conversation_id': conversation['conversation_id'], 'evaluation_id': eid})
                job['completed'] = len(job['results']) + len(job['errors'])
                job['updated'] = now()
                db.execute('UPDATE jobs SET body=? WHERE id=?', (json.dumps(job, ensure_ascii=False), job_id))
        return self.evaluation(eid)

    def save_job(self, job):
        with self.db() as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body',
                       (job['id'], json.dumps(job, ensure_ascii=False), job['created']))

    def job(self, jid):
        if not isinstance(jid, str) or not jid:
            raise ValueError('Укажите ID задачи.')
        with self.db() as db:
            row = db.execute('SELECT body FROM jobs WHERE id=?', (jid,)).fetchone()
            if not row: raise ValueError('Задача не найдена.')
            return json.loads(row['body'])

    def recent_jobs(self, limit=20):
        with self.db() as db:
            return [json.loads(row['body']) for row in db.execute(
                'SELECT body FROM jobs ORDER BY rowid DESC LIMIT ?', (limit,))]

    def interrupt_jobs(self):
        # A process restart cannot resume an in-flight provider request safely.
        with self.db() as db:
            for row in db.execute('SELECT id,body FROM jobs').fetchall():
                job = json.loads(row['body'])
                if job['status'] in ['queued', 'running', 'cancelling']:
                    job.update(status='interrupted', message='Сервер перезапущен. Можно продолжить незавершённые диалоги.', updated=now())
                    db.execute('UPDATE jobs SET body=? WHERE id=?', (json.dumps(job, ensure_ascii=False), row['id']))

    def evaluation(self, eid):
        with self.db() as db:
            row = db.execute('SELECT body FROM evaluations WHERE id=?', (eid,)).fetchone()
            if not row: raise ValueError('Оценка не найдена.')
            result = json.loads(row['body'])
            appeals = [dict(a) for a in db.execute('SELECT * FROM appeals WHERE evaluation_id=? ORDER BY rowid', (eid,))]
        final = copy.deepcopy(result['assessment'])
        for a in appeals:
            if a['status'] == 'resolved':
                final['criteria'][a['criterion_id']]['answer'] = a['answer']
                final['criteria'][a['criterion_id']]['reviewer_reason'] = a['reason']
        return {**result, 'appeals': appeals, 'final_assessment': final,
                'final_score': calculate(result['checklist'], final)}

    def history(self, cid):
        with self.db() as db:
            return [{'id': r['id'], 'created': r['created']} for r in db.execute(
                'SELECT id,created FROM evaluations WHERE conversation_id=? ORDER BY rowid DESC', (cid,))]

    def appeal(self, eid, cid, comment):
        evaluation = self.evaluation(eid)
        if cid not in evaluation['assessment']['criteria']:
            raise ValueError('Критерий не найден.')
        if not isinstance(comment, str) or not 1 <= len(comment.strip()) <= 5000:
            raise ValueError('Комментарий должен содержать 1–5000 символов.')
        aid = str(uuid.uuid4())
        with self.db() as db:
            if db.execute('SELECT id FROM appeals WHERE evaluation_id=? AND criterion_id=? AND status=?',
                          (eid, cid, 'open')).fetchone():
                raise ValueError('По критерию уже есть открытая апелляция.')
            db.execute('INSERT INTO appeals VALUES (?,?,?,?,?,?,?,?,?)',
                       (aid, eid, cid, comment.strip(), 'open', None, None, now(), None))
        return aid

    def appeals(self):
        with self.db() as db:
            rows = [dict(r) for r in db.execute('SELECT * FROM appeals ORDER BY rowid DESC')]
            for r in rows:
                e = db.execute('SELECT conversation_id FROM evaluations WHERE id=?', (r['evaluation_id'],)).fetchone()
                r['conversation_id'] = e['conversation_id']
        return rows

    def resolve(self, aid, answer, reason):
        with self.db() as db:
            a = db.execute('SELECT * FROM appeals WHERE id=?', (aid,)).fetchone()
        if not a: raise ValueError('Апелляция не найдена.')
        evaluation = self.evaluation(a['evaluation_id'])
        criterion = next(c for c in evaluation['checklist']['criteria'] if c['id'] == a['criterion_id'])
        if answer not in choices(criterion) or not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 5000:
            raise ValueError('Выберите допустимую оценку и укажите причину решения.')
        with self.db() as db:
            cursor = db.execute('UPDATE appeals SET status=?,answer=?,reason=?,resolved=? WHERE id=? AND status=?',
                                ('resolved', answer, reason.strip(), now(), aid, 'open'))
            if cursor.rowcount != 1: raise ValueError('Апелляция уже рассмотрена.')
        return self.evaluation(a['evaluation_id'])
