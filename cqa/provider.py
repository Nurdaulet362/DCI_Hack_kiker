"""Small HTTP adapters. Credentials stay on the server; no SDK dependency."""
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

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
    return {'provider': provider, 'base_url': base, 'model': model, 'key': key,
            'ready': bool(base and model and (key or provider == 'ollama')),
            'review': os.getenv('AI_REVIEW', 'true').lower() == 'true',
            'reasoning_effort': os.getenv('AI_REASONING_EFFORT', ''),
            'context_window': os.getenv('AI_CONTEXT_WINDOW', '32768'),
            'think': os.getenv('AI_THINK', 'false').lower() == 'true',
            'max_output_tokens': os.getenv('AI_MAX_OUTPUT_TOKENS', '') or '8192',
            'timeout_seconds': os.getenv('AI_TIMEOUT_SECONDS', '') or '180'}


def public_config():
    c = config()
    return {k: c[k] for k in ['provider', 'model', 'ready', 'review']}


def retry_after_seconds(value):
    """Return the server's minimum wait, without retaining response text."""
    if not value:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None: date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    if not math.isfinite(seconds): return None
    return max(0, math.ceil(seconds))


class Provider:
    def __init__(self):
        self.cfg = config()
        if not self.cfg['ready']:
            key_hint = '' if self.cfg['provider'] == 'ollama' else ' и AI_API_KEY'
            raise ProviderError(f'AI не настроен. Заполните AI_PROVIDER, AI_BASE_URL, AI_MODEL{key_hint} в .env и перезапустите сервер.')
        if self.cfg['provider'] not in ['openai', 'openai_compatible', 'ollama']:
            raise ProviderError('Поддерживаются openai, openai_compatible и ollama.')
        if self.cfg['reasoning_effort'] not in ['', 'none', 'minimal', 'low', 'medium', 'high']:
            raise ProviderError('AI_REASONING_EFFORT: допустимы none, minimal, low, medium или high.')
        for field, env, minimum, maximum in [('context_window', 'AI_CONTEXT_WINDOW', 4096, 131072),('max_output_tokens', 'AI_MAX_OUTPUT_TOKENS', 1024, 32768),
                                             ('timeout_seconds', 'AI_TIMEOUT_SECONDS', 10, 300)]:
            try:
                value = int(self.cfg[field])
                if not minimum <= value <= maximum: raise ValueError()
                self.cfg[field] = value
            except (TypeError, ValueError):
                raise ProviderError(f'{env}: укажите целое число от {minimum} до {maximum}.') from None
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
                    'max_output_tokens': c['max_output_tokens'],
                    'text': {'format': {'type': 'json_schema', 'name': 'assessment',
                                        'strict': True, 'schema': schema}}}
            if c['reasoning_effort']: body['reasoning'] = {'effort': c['reasoning_effort']}
        elif c['provider'] == 'ollama':
            endpoint = '/api/chat'
            # Ollama's constrained decoder does not replace a readable output contract.
            messages[0]['content'] += '\nВерни JSON по этой схеме:\n' + json.dumps(schema, ensure_ascii=False)
            body = {'model': c['model'], 'messages': messages, 'stream': False,
                    'format': schema, 'think': c.get('think', False),
                    'options': {'temperature': 0, 'num_ctx': c.get('context_window', 32768),
                                'num_predict': c['max_output_tokens']}}
        else:
            endpoint = '/chat/completions'
            body = {'model': c['model'], 'messages': messages,
                    'max_completion_tokens': c['max_output_tokens'],
                    'response_format': {'type': 'json_schema', 'json_schema': {
                        'name': 'assessment', 'strict': True, 'schema': schema}}}
            if c['reasoning_effort']: body['reasoning_effort'] = c['reasoning_effort']
        headers = {'Content-Type': 'application/json'}
        if c['key']: headers['Authorization'] = 'Bearer ' + c['key']
        started = time.monotonic()
        for attempt in range(3):
            try:
                req = urllib.request.Request(c['base_url'] + endpoint, json.dumps(body).encode(), headers)
                with open_api(req, timeout=c['timeout_seconds']) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise ProviderError('Ответ AI API превышает 4 МБ.')
                result = json.loads(raw)
                if not isinstance(result, dict): raise ValueError()
                break
            except urllib.error.HTTPError as e:
                code = e.code
                wait = retry_after_seconds(e.headers.get('Retry-After')) if e.headers else None
                e.close()  # Never expose a gateway's response body or echoed credentials.
                if code in [429, 500, 502, 503, 504] and attempt < 2:
                    delay = wait if wait is not None else (5 if code == 429 else 2) * (2 ** attempt)
                    if delay <= 30:
                        time.sleep(delay)
                        continue
                if code == 429:
                    suffix = f' Повторите не раньше чем через {wait} сек.' if wait else ' Проверьте квоту и повторите позже.'
                    raise ProviderError('AI API: лимит запросов (HTTP 429).' + suffix) from None
                if code in [401, 403]:
                    raise ProviderError(f'AI API: доступ отклонён (HTTP {code}). Проверьте ключ, права и доступность модели.') from None
                if code == 404:
                    raise ProviderError('AI API: модель или адрес API недоступны (HTTP 404). Проверьте AI_MODEL и AI_BASE_URL.') from None
                raise ProviderError(f'AI API: HTTP {code}. Проверьте доступность API и поддержку JSON Schema.') from None
            except TimeoutError:
                raise ProviderError(f'AI API: превышено время ожидания {c["timeout_seconds"]} сек. Результат не сохранён; повторите позже или выберите более быстрый режим.') from None
            except urllib.error.URLError as e:
                if isinstance(e.reason, TimeoutError):
                    raise ProviderError(f'AI API: превышено время ожидания {c["timeout_seconds"]} сек.') from None
                raise ProviderError('AI API: ошибка соединения. Проверьте сеть и доступность провайдера.') from None
            except (OSError, ValueError):
                raise ProviderError('AI API недоступен или вернул некорректный ответ.') from None
        try:
            if c['provider'] == 'openai':
                truncated = result.get('status') == 'incomplete'
                content = ''.join(part.get('text', '') for item in result.get('output', [])
                                  for part in item.get('content', []) if part.get('type') == 'output_text')
                usage = result.get('usage', {})
                tokens_in, tokens_out = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
            elif c['provider'] == 'ollama':
                truncated = result.get('done_reason') == 'length'
                content = result.get('message', {}).get('content', '')
                tokens_in, tokens_out = result.get('prompt_eval_count', 0), result.get('eval_count', 0)
            else:
                choice = result.get('choices', [{}])[0]
                truncated = choice.get('finish_reason') == 'length'
                content = choice.get('message', {}).get('content', '')
                usage = result.get('usage', {})
                tokens_in, tokens_out = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
            if truncated:
                raise ProviderError('Ответ AI обрезан лимитом токенов. Увеличьте AI_MAX_OUTPUT_TOKENS или сократите чек-лист; результат не сохранён.')
            parsed = json.loads(content)
            if not isinstance(parsed, dict): raise ValueError()
            if any(type(v) is not int or v < 0 for v in [tokens_in, tokens_out]): raise ValueError()
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise ProviderError('Модель не вернула JSON-оценку (отказ, некорректный ответ или несовместимый API).') from None
        cost = None
        try:
            prices = [float(os.environ[k]) for k in ['AI_INPUT_PRICE', 'AI_OUTPUT_PRICE']]
            if all(math.isfinite(p) and p >= 0 for p in prices):
                cost = (tokens_in * prices[0] + tokens_out * prices[1]) / 1000000
        except (KeyError, ValueError):
            pass
        return parsed, {'input_tokens': tokens_in, 'output_tokens': tokens_out,
                        'seconds': round(time.monotonic()-started, 2), 'cost': cost,
                        'http_attempts': attempt + 1}
