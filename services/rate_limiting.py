import time
import re
from groq import Groq, RateLimitError
from core.config import GROQ_TPM_LIMIT, GROQ_TPM_BUFFER, TOKEN_WINDOW, GROQ_API_KEY


class TokenBudget:
    """Tracks and enforces TPM (tokens per minute) rate limits."""

    def __init__(self, tpm_limit: int, tpm_buffer: float, window: int):
        self._limit        = tpm_limit * tpm_buffer
        self._window       = window
        self._tokens_used  = 0
        self._window_start = time.time()

    def _reset_if_needed(self) -> None:
        elapsed = time.time() - self._window_start
        if elapsed >= self._window:
            print(f"[INFO] Token window reset. ({elapsed:.1f}s elapsed)")
            self._tokens_used  = 0
            self._window_start = time.time()

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """Sleep until window resets if budget is nearly exhausted."""
        self._reset_if_needed()

        if self._tokens_used + estimated_tokens >= self._limit:
            wait = max(0, self._window - (time.time() - self._window_start) + 2)
            print(f"[RATE LIMIT] Budget near limit ({self._tokens_used:,} used). "
                  f"Waiting {wait:.1f}s...")
            time.sleep(wait)
            self._tokens_used  = 0
            self._window_start = time.time()

    def record(self, tokens: int) -> None:
        """Record actual tokens used after a successful call."""
        self._tokens_used += tokens
        print(f"[TOKEN] Used {tokens} | "
              f"Window total: {self._tokens_used:,}/{self._limit:,.0f}")

    def reset(self) -> None:
        self._tokens_used  = 0
        self._window_start = time.time()


class GroqClient:
    """Groq LLM client with built-in rate limiting and retry logic."""

    def __init__(
        self,
        api_key: str         = GROQ_API_KEY,
        tpm_limit: int       = GROQ_TPM_LIMIT,
        tpm_buffer: float    = GROQ_TPM_BUFFER,
        token_window: int    = TOKEN_WINDOW,
        max_retries: int     = 5
    ):
        self._client      = Groq(api_key=api_key)
        self._budget      = TokenBudget(tpm_limit, tpm_buffer, token_window)
        self._max_retries = max_retries

    def complete(self, messages: list, model: str, response_format=None) -> object:
        """
        Call Groq chat completions with proactive rate limiting and
        exponential backoff retry on RateLimitError.
        """
        estimated = sum(len(m["content"]) // 4 for m in messages)
        self._budget.wait_if_needed(estimated)

        delay = 5
        for attempt in range(self._max_retries):
            try:
                kwargs = {"model": model, "messages": messages}
                if response_format:
                    kwargs["response_format"] = response_format

                response = self._client.chat.completions.create(**kwargs)
                self._budget.record(response.usage.total_tokens)
                return response

            except RateLimitError as e:
                if attempt == self._max_retries - 1:
                    raise

                wait = self._parse_retry_after(str(e), fallback=delay)
                print(f"[RATE LIMIT] Attempt {attempt + 1}. Waiting {wait}s...")
                time.sleep(wait)
                self._budget.reset()
                delay *= 2

    @staticmethod
    def _parse_retry_after(error_msg: str, fallback: int) -> int:
        match = re.search(
            r"(?:try again in|retry after)\s*([\d.]+)",
            error_msg,
            re.IGNORECASE
        )
        return int(float(match.group(1))) + 1 if match else fallback