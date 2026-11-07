"""Small HTTP adapters. Credentials stay on the server; no SDK dependency."""
import json
import os
import time
import urllib.error
import urllib.request

from . import settings  # loads .env before reading configuration
from .network import open_api, validate_url


class ProviderError(RuntimeError):
    pass


def config():
    provider = os.getenv('AI_PROVIDER', 'openai_compatible')
    base = os.getenv('AI_BASE_URL', '').rstrip('/')
    if provider == 'openai': base = base or 'https://api.openai.com/v1'
    if provider == 'ollama': base = base or 'http://127.0.0.1:11434'
    model = os.getenv('AI_MODEL', '')
    key = os.getenv('AI_API_KEY', '') or (os.getenv('OPENAI_API_KEY', '') if provider == 'openai' else '')
    ready = bool(base and model and (key or provider == 'ollama'))
    return {'provider': provider, 'base_url': base, 'model': model, 'key': key, 'ready': ready,
            'review': os.getenv('AI_REVIEW', 'true').lower() == 'true',
            'reasoning_effort': os.getenv('AI_REASONING_EFFORT', '')}


def public_config():
    c = config()
    return {k: c[k] for k in ['provider', 'model', 'ready', 'review']}


class Provider:
    def __init__(self):
        self.cfg = config()
        if not self.cfg['ready']:
            raise ProviderError('AI не подключён. Укажите AI_PROVIDER, AI_BASE_URL, AI_MODEL и AI_API_KEY в .env, затем перезапустите сервер.')
        if self.cfg['provider'] not in ['openai', 'openai_compatible', 'ollama']:
            raise ProviderError('Поддерживаются openai, openai_compatible и ollama.')
        if self.cfg['reasoning_effort'] not in ['', 'none', 'minimal', 'low', 'medium', 'high']:
            raise ProviderError('AI_REASONING_EFFORT: допустимы none, minimal, low, medium или high.')
        try:
            validate_url(self.cfg['base_url'])
        except ValueError as e:
            raise ProviderError(str(e)) from None

    def complete(self, system, user, schema):
        c = self.cfg
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        if c['provider'] == 'openai':
            endpoint = '/responses'
            body = {'model': c['model'], 'instructions': system,
                    'input': [{'role': 'user', 'content': user}], 'store': False,
                    'text': {'format': {'type': 'json_schema', 'name': 'assessment',
                                        'strict': True, 'schema': schema}}}
        elif c['provider'] == 'ollama':
            endpoint = '/api/chat'
            body = {'model': c['model'], 'messages': messages, 'stream': False,
                    'format': schema, 'options': {'temperature': 0}}
        else:
            endpoint = '/chat/completions'
            body = {'model': c['model'], 'messages': messages,
                    'response_format': {'type': 'json_schema', 'json_schema': {
                        'name': 'assessment', 'strict': True, 'schema': schema}}}
            if c['reasoning_effort']:
                body['reasoning_effort'] = c['reasoning_effort']
        headers = {'Content-Type': 'application/json'}
        if c['key']: headers['Authorization'] = 'Bearer ' + c['key']
        started = time.monotonic()
        for attempt in range(3):
            try:
                req = urllib.request.Request(c['base_url'] + endpoint,
                                             json.dumps(body).encode(), headers)
                with open_api(req, timeout=180) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise ProviderError('Ответ AI API превышает 4 МБ.')
                result = json.loads(raw)
                break
            except urllib.error.HTTPError as e:
                e.close()
                # Do not expose response bodies: some gateways echo credentials and requests.
                if e.code in [429, 500, 502, 503, 504] and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise ProviderError(f'AI API: HTTP {e.code}. Проверьте модель, ключ, баланс и поддержку JSON Schema.') from None
            except (TimeoutError, OSError, ValueError):
                raise ProviderError('AI API недоступен или вернул некорректный ответ.') from None
        if c['provider'] == 'openai':
            content = ''.join(part.get('text', '') for item in result.get('output', [])
                              for part in item.get('content', []) if part.get('type') == 'output_text')
            usage = result.get('usage', {})
            tokens_in, tokens_out = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
        elif c['provider'] == 'ollama':
            content = result.get('message', {}).get('content', '')
            tokens_in, tokens_out = result.get('prompt_eval_count', 0), result.get('eval_count', 0)
        else:
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            usage = result.get('usage', {})
            tokens_in, tokens_out = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            raise ProviderError('Модель не вернула JSON-оценку (отказ, обрезанный ответ или несовместимый API).') from None
        cost = None
        try:
            cost = (tokens_in * float(os.environ['AI_INPUT_PRICE']) +
                    tokens_out * float(os.environ['AI_OUTPUT_PRICE'])) / 1000000
        except (KeyError, ValueError):
            pass
        return parsed, {'input_tokens': tokens_in, 'output_tokens': tokens_out,
                        'seconds': round(time.monotonic()-started, 2), 'cost': cost}
