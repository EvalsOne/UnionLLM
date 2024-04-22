import unittest
from unittest.mock import patch
from cnlitellm.providers.moonshot import MoonshotAIProvider
from cnlitellm.exceptions import CNLiteLLMError

class TestMoonshotAIProvider(unittest.TestCase):

    def setUp(self):
        self.provider = MoonshotAIProvider(api_key='test_api_key')

    @patch('cnlitellm.providers.moonshot.ZhipuAI')
    def test_completion_success(self, mock_zhipu):
        mock_zhipu.return_value.chat.completions.create.return_value = 'Test response'
        response = self.provider.completion(
            model='test-model',
            messages=['Hello, world!'],
            max_tokens=50
        )
        self.assertEqual(response, 'Test response')

    @patch('cnlitellm.providers.moonshot.ZhipuAI')
    def test_completion_failure_missing_model(self, mock_zhipu):
        with self.assertRaises(CNLiteLLMError) as context:
            self.provider.completion(
                model=None,
                messages=['Hello, world!']
            )
        self.assertEqual(context.exception.status_code, 422)
        self.assertEqual(context.exception.message, 'Missing model or messages')

    @patch('cnlitellm.providers.moonshot.ZhipuAI')
    def test_completion_api_error(self, mock_zhipu):
        mock_zhipu.return_value.chat.completions.create.side_effect = Exception("API error")
        with self.assertRaises(CNLiteLLMError) as context:
            self.provider.completion(
                model='test-model',
                messages=['Hello, world!']
            )
        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(context.exception.message, 'API error')

if __name__ == '__main__':
    unittest.main()
