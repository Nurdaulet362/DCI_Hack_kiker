import io
import json
import os
import unittest
from unittest.mock import patch

from cqa.provider import Provider, ProviderError


class ProviderTests(unittest.TestCase):
    def test_compatible_structured_output_and_optional_reasoning(self):
        env = {'AI_PROVIDER': 'openai_compatible', 'AI_BASE_URL': 'https://example.test/v1',
               'AI_MODEL': 'test-model', 'AI_API_KEY': 'test-only'}
        schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
        for effort in ['', 'low']:
            with self.subTest(effort=effort), patch.dict(os.environ, {**env, 'AI_REASONING_EFFORT': effort}):
                reply = {'choices': [{'message': {'content': '{"ok": true}'}}],
                         'usage': {'prompt_tokens': 10, 'completion_tokens': 4}}
                with patch('cqa.provider.open_api', return_value=io.BytesIO(json.dumps(reply).encode())) as send:
                    answer, usage = Provider().complete('audit', 'dialogue', schema)
                request = send.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(payload['response_format']['json_schema']['schema'], schema)
                self.assertEqual(request.get_header('Authorization'), 'Bearer test-only')
                self.assertNotIn('test-only', request.full_url)
                self.assertEqual(answer, {'ok': True})
                self.assertEqual(usage['input_tokens'], 10)
                if effort:
                    self.assertEqual(payload['reasoning_effort'], effort)
                else:
                    self.assertNotIn('reasoning_effort', payload)

    def test_invalid_reasoning_fails_before_network(self):
        with patch.dict(os.environ, {'AI_PROVIDER': 'openai_compatible',
                'AI_BASE_URL': 'https://example.test/v1', 'AI_MODEL': 'test-model',
                'AI_API_KEY': 'test-only', 'AI_REASONING_EFFORT': 'invalid'}), \
                patch('cqa.provider.open_api') as send:
            with self.assertRaises(ProviderError):
                Provider()
            send.assert_not_called()

class ProviderFailureTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'AI_PROVIDER': 'openai_compatible',
            'AI_BASE_URL': 'https://example.test/v1', 'AI_MODEL': 'fixture',
            'AI_API_KEY': 'secret-test-key', 'AI_REASONING_EFFORT': 'low',
            'AI_MAX_OUTPUT_TOKENS': '8192', 'AI_TIMEOUT_SECONDS': '30'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_long_retry_after_does_not_retry_early(self):
        import urllib.error
        err = urllib.error.HTTPError('https://example.test', 429, 'limited',
                                     {'Retry-After': '120'}, io.BytesIO(b'secret-test-key'))
        with patch('cqa.provider.open_api', side_effect=err) as send, patch('cqa.provider.time.sleep') as sleep:
            with self.assertRaises(ProviderError) as caught:
                Provider().complete('system', 'user', {})
            self.assertIn('120', str(caught.exception))
            self.assertNotIn('secret-test-key', str(caught.exception))
            self.assertEqual(send.call_count, 1)
            sleep.assert_not_called()

    def test_short_retry_after_is_honored_then_succeeds(self):
        import urllib.error
        err = urllib.error.HTTPError('https://example.test', 429, 'limited', {'Retry-After': '3'}, io.BytesIO())
        reply = {'choices': [{'message': {'content': '{"ok": true}'}}]}
        with patch('cqa.provider.open_api', side_effect=[err, io.BytesIO(json.dumps(reply).encode())]), \
                patch('cqa.provider.time.sleep') as sleep:
            result, usage = Provider().complete('system', 'user', {})
        sleep.assert_called_once_with(3)
        self.assertTrue(result['ok'])
        self.assertEqual(usage['http_attempts'], 2)

    def test_truncated_valid_json_is_rejected(self):
        reply = {'choices': [{'finish_reason': 'length', 'message': {'content': '{"ok": true}'}}]}
        with patch('cqa.provider.open_api', return_value=io.BytesIO(json.dumps(reply).encode())):
            with self.assertRaisesRegex(ProviderError, 'обрезан'):
                Provider().complete('system', 'user', {})

    def test_timeout_is_explicit_and_bounded(self):
        with patch('cqa.provider.open_api', side_effect=TimeoutError()) as send:
            with self.assertRaisesRegex(ProviderError, '30 сек'):
                Provider().complete('system', 'user', {})
        self.assertEqual(send.call_args.kwargs['timeout'], 30)
        self.assertEqual(send.call_count, 1)

    def test_auth_body_never_leaks(self):
        import urllib.error
        err = urllib.error.HTTPError('https://example.test', 401, 'secret-test-key', {},
                                     io.BytesIO(b'secret-test-key'))
        with patch('cqa.provider.open_api', side_effect=err):
            with self.assertRaises(ProviderError) as caught:
                Provider().complete('system', 'user', {})
        self.assertIn('доступ отклонён', str(caught.exception))
        self.assertNotIn('secret-test-key', str(caught.exception))

    def test_empty_choices_is_controlled_error(self):
        with patch('cqa.provider.open_api', return_value=io.BytesIO(b'{"choices":[]}')):
            with self.assertRaises(ProviderError):
                Provider().complete('system', 'user', {})

    def test_output_limit_validated_before_request(self):
        with patch.dict(os.environ, {'AI_MAX_OUTPUT_TOKENS': '0'}):
            with self.assertRaisesRegex(ProviderError, 'AI_MAX_OUTPUT_TOKENS'):
                Provider()
