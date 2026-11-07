import json
import os
import urllib.error
import urllib.parse
import urllib.request

from . import settings


def fetch_dataset(group_id='DEV'):
    if not isinstance(group_id, str) or not group_id or len(group_id) > 100:
        raise ValueError('Некорректный group_id.')
    url = os.getenv('DATASET_API_URL', '')
    if not url: raise ValueError('Добавьте DATASET_API_URL в .env и перезапустите сервер.')
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    params['group_id'] = [group_id]
    url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(params, doseq=True)))
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            raw = r.read(20*1024*1024+1)
        if len(raw) > 20*1024*1024: raise ValueError('Ответ API превышает 20 МБ.')
        return json.loads(raw)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        # A signed URL is a credential. Never echo it in logs/errors or the browser.
        raise ValueError('Не удалось получить набор. Проверьте доступность API, срок ссылки и group_id.') from None
