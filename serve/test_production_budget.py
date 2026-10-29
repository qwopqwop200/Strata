"""Model-free checks for the opt-in production budget and honest schema policy."""
import json
import unittest
import urllib.request

from serve import frontend, structured
from serve.server import CTX_SLACK, Service, openai_collect
from serve.test_dsv41 import ApiBase, DsEngine, TEMPLATE, mini_tokenizer
from serve.frontend import ChatTemplate


class Production(ApiBase):
    def test_numeric_all_100_through_http_and_template(self):
        self.start('Compute.</think>2')
        for n in range(1, 101):
            with self.subTest(n=n):
                stream = n % 2 == 0
                code, reply = self.post('/v1/chat/completions', {
                    'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 400,
                    'reasoning_effort': n, 'stream': stream}, raw=stream)
                self.assertEqual(code, 200)
                self.assertIn(f'Reasoning Effort: {n} (range 1-100', self.prompt())
        for value in (0, 101, 1.5, 62.0, True, '1.5'):
            self.assertEqual(self.chat(reasoning_effort=value)[0], 400, value)

    def test_budget_nominal_and_effective_in_both_response_modes(self):
        self.start('Compute.</think>2')
        self.svc.default_max_tokens = 262144
        self.svc.context_slack_tokens = 0
        self.svc.report_output_budget = True
        with urllib.request.urlopen(self.base + '/props') as response:
            props = json.load(response)
        self.assertEqual(props['default_generation_settings']['params']['n_predict'], 262144)
        for stream in (False, True):
            code, reply = self.post('/v1/chat/completions', {
                'messages': [{'role': 'user', 'content': '1+1?'}], 'stream': stream}, raw=stream)
            self.assertEqual(code, 200)
            if stream:
                reply = openai_collect([json.loads(x[6:]) for x in reply.splitlines() if x.startswith('data: {')])
            budget = reply['output_budget']
            self.assertEqual(budget['requested_max_tokens'], 262144)
            self.assertEqual(budget['effective_max_tokens'] + budget['encoded_input_tokens'], self.engine.max_context)
            self.assertFalse(budget['input_truncated'])

    def test_strict_schema_refusal_is_before_generation(self):
        self.start('Compute.</think>{"ok":true}')
        self.svc.require_strict_schema_constraint = True
        for stream in (False, True):
            code, reply = self.chat(stream=stream, response_format={'type': 'json_schema', 'json_schema': {
                'name': 'proof', 'strict': True, 'schema': {'type': 'object', 'properties': {
                    'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False}}})
            self.assertEqual(code, 400)
            self.assertIn('strict constrained decoding is unsupported', reply['error']['message'])
            self.assertIsNone(self.engine.last_sampling)

    def test_legacy_defaults_unchanged(self):
        self.start('Compute.</think>2')
        self.assertEqual(self.svc.default_max_tokens, 0)
        self.assertEqual(self.svc.context_slack_tokens, CTX_SLACK)
        self.assertFalse(self.svc.require_strict_schema_constraint)
        self.assertFalse(self.svc.report_output_budget)


if __name__ == '__main__':
    unittest.main()
