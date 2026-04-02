"""
API call logging for hosted LLM backends.
"""

import time
from typing import Dict, List, Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from logger import get_logger

log = get_logger(__name__)


def _is_hosted_backend(backend: str) -> bool:
    """Check if the backend is a hosted (API-based) service."""
    return backend.lower() in ("openai", "gemini")


class APICallLogger(BaseCallbackHandler):
    """Callback handler to log LLM API calls for hosted backends."""
    
    def __init__(self, backend: str):
        self.backend = backend
        self.start_time: Optional[float] = None
        self.model_name: str = ""
        self.input_tokens: int = 0
        self.output_tokens: int = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts running."""
        self.start_time = time.time()
        self.model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
        log.info("LLM API call started: backend=%s, model=%s", self.backend, self.model_name)
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Extract token usage if available
        if response and hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.input_tokens = token_usage.get("prompt_tokens", 0)
            self.output_tokens = token_usage.get("completion_tokens", 0)
        
        log.info(
            "LLM API call completed: backend=%s, model=%s, duration=%.2fs, input_tokens=%d, output_tokens=%d",
            self.backend, self.model_name, duration, self.input_tokens, self.output_tokens
        )
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error."""
        duration = time.time() - self.start_time if self.start_time else 0
        log.error("LLM API call failed: backend=%s, model=%s, duration=%.2fs, error=%s",
                  self.backend, self.model_name, duration, str(error))
