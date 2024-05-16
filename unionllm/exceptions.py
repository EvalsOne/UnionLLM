class UnionLLMError(Exception):
    """Base exception class for UnionLLM."""
    pass

class ProviderError(UnionLLMError):
    """Exception raised when there is an issue with the model provider."""
    pass

class APICallError(UnionLLMError):
    """Exception raised when an API call fails."""
    pass