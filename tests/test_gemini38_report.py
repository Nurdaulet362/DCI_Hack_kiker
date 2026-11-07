"""Offline, synthetic test fixtures. Never call the Gemini API."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.gemini38_report import summarize, percentile


class Gemini38ReportTests(unittest.TestCase):
    def test_nonempty_and_latency_and_token_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'run.json'
            filename.write_text(json.dumps([
                {'conversation_id': 'a',
                 'assessment': {'criteria': {'contact': {'answer': 'yes'}}}, 'score': {'total': 100},
                 'review_changes': ['contact'],
                 'usage': {'seconds': 2.0, 'input_tokens': 100, 'output_tokens': 50,
                           'per_call': [{'stage': 'initial', 'http_attempts': 1},
                                        {'stage': 'review', 'http_attempts': 2}]}},
                {'conversation_id': 'b', 'error': 'timeout'},
                {'conversation_id': 'c',
                 'assessment': {'criteria': {'contact': {'answer': 'yes'}}}, 'score': {'total': 100},
                 'review_changes': [],
                 'usage': {'seconds': 4.0, 'input_tokens': 200, 'output_tokens': 100,
                           'per_call': [{'stage': 'initial'}]}},
            ]), encoding='utf-8')
            summary = summarize(filename, 1.0, 2.0)
            self.assertEqual((summary['selected_records'], summary['successful'],
                              summary['failed_or_incomplete']), (3, 2, 1))
            self.assertEqual(summary['token_usage']['total_tokens'], 450)
            self.assertEqual(summary['provider_latency_seconds']['median_p50'], 3)
            self.assertEqual(summary['provider_latency_seconds']['p95_nearest_rank'], 4)
            self.assertEqual(summary['total_http_attempts_if_reported'], 4)
            self.assertEqual(summary['calls_by_stage'], {'initial': 2, 'review': 1})
            self.assertEqual(summary['review_changed_records'], 1)
            self.assertEqual(summary['estimated_cost_usd'], .0006)

    def test_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'run.json'
            filename.write_text('[{"conversation_id":"a"},{"conversation_id":"a"}]')
            with self.assertRaises(ValueError):
                summarize(filename)

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'run.json'
            filename.write_text('[]')
            summary = summarize(filename)
            self.assertEqual(summary['selected_records'], 0)
            self.assertIsNone(summary['provider_latency_seconds']['median_p50'])
            self.assertIsNone(percentile([], .95))
