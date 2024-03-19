class CNLiteLLMError(Exception):
    """Base exception class for CNLiteLLM."""
    pass

class ProviderError(CNLiteLLMError):
    """Exception raised when there is an issue with the model provider."""
    pass

class APICallError(CNLiteLLMError):
    """Exception raised when an API call fails."""
    pass