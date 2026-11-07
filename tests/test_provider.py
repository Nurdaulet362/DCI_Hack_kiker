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
