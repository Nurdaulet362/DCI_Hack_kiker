"""No network calls: guardrails for the optional Gemini 3.8 experiment."""
import os
import unittest
from unittest.mock import patch

from scripts.gemini38_experiment import PROFILE, activate_profile, select_ids


class Gemini38ExperimentTests(unittest.TestCase):
    def test_profile_is_gemini_38_and_does_not_replace_secret(self):
        with patch.dict(os.environ, {'AI_API_KEY': 'keep-this-secret',
                                     'AI_INPUT_PRICE': '999', 'AI_OUTPUT_PRICE': '999'}):
            activate_profile(review=False)
            self.assertEqual(os.environ['AI_MODEL'], 'gemini-3.8-flash')
            self.assertEqual(os.environ['AI_BASE_URL'], PROFILE['AI_BASE_URL'])
            self.assertEqual(os.environ['AI_REASONING_EFFORT'], 'low')
            self.assertEqual(os.environ['AI_REVIEW'], 'false')
            self.assertEqual(os.environ['AI_API_KEY'], 'keep-this-secret')
            self.assertNotIn('AI_INPUT_PRICE', os.environ)
            self.assertNotIn('AI_OUTPUT_PRICE', os.environ)

    def test_initial_review_mode_remains_two_passes(self):
        with patch.dict(os.environ, {}):
            activate_profile(review=True)
            self.assertEqual(os.environ['AI_REVIEW'], 'true')

    def test_id_selection_and_invalid_inputs(self):
        rows = [{'conversation_id': cid} for cid in ('a', 'b', 'c')]
        self.assertEqual(select_ids(rows, None, 2, False), ['a', 'b'])
        self.assertEqual(select_ids(rows, None, 1, True), ['a', 'b', 'c'])
        self.assertEqual(select_ids(rows, ['c', 'a'], 1, False), ['c', 'a'])
        with self.assertRaises(ValueError):
            select_ids(rows, ['a', 'a'], 2, False)
        with self.assertRaises(ValueError):
            select_ids(rows, ['unknown'], 2, False)
        with self.assertRaises(ValueError):
            select_ids(rows, None, 0, False)


if __name__ == '__main__':
    unittest.main()
