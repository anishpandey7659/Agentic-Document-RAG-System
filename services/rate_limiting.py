import time
from groq import Groq, RateLimitError, APIStatusError
from config import GROQ_TPM_LIMIT, GROQ_TPM_BUFFER, TOKEN_WINDOW ,GROQ_API_KEY
import re

groq_client = Groq(api_key=GROQ_API_KEY)  

_token_tracker = {
    "tokens_used": 0,
    "window_start": time.time()
}


def _reset_token_window_if_needed():
    """Reset token counter if 60s window has passed."""
    now = time.time()
    elapsed = now - _token_tracker["window_start"]

    if elapsed >= TOKEN_WINDOW:
        print(f"[INFO] Token window reset. ({elapsed:.1f}s elapsed)")
        _token_tracker["tokens_used"] = 0
        _token_tracker["window_start"] = now


def _wait_if_token_limit_near(tokens_about_to_use: int):
    """
    If adding the next call's tokens would exceed 85% of TPM limit,
    sleep until the 60s window resets.
    """
    _reset_token_window_if_needed()

    projected = _token_tracker["tokens_used"] + tokens_about_to_use
    limit      = GROQ_TPM_LIMIT * GROQ_TPM_BUFFER

    if projected >= limit:
        elapsed   = time.time() - _token_tracker["window_start"]
        wait_time = max(0, TOKEN_WINDOW - elapsed + 2)   # +2s buffer
        print(f"[RATE LIMIT] Token budget near limit ({_token_tracker['tokens_used']:,} used). "
              f"Waiting {wait_time:.1f}s for window reset...")
        time.sleep(wait_time)

        # Reset after waiting
        _token_tracker["tokens_used"] = 0
        _token_tracker["window_start"] = time.time()


def _call_groq_with_retry(messages, model, response_format=None, max_retries=5):
    """
    Groq call with:
    - Proactive token tracking (pauses before hitting limit)
    - Retry + exponential backoff as safety net
    """
    # Estimate tokens about to be used (rough: 1 token ≈ 4 chars)
    estimated_tokens = sum(len(m["content"]) // 4 for m in messages)
    _wait_if_token_limit_near(estimated_tokens)

    delay = 5
    for attempt in range(max_retries):
        try:
            kwargs = {"model": model, "messages": messages}
            if response_format:
                kwargs["response_format"] = response_format

            response = groq_client.chat.completions.create(**kwargs)

            # Track actual tokens used from response
            actual_tokens = response.usage.total_tokens
            _token_tracker["tokens_used"] += actual_tokens
            print(f"[TOKEN] Used {actual_tokens} tokens this call | "
                  f"Total this window: {_token_tracker['tokens_used']:,}/{GROQ_TPM_LIMIT:,}")

            return response

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            wait_time = _parse_retry_after(str(e), fallback=delay)
            print(f"[RATE LIMIT] Hit on attempt {attempt+1}. Waiting {wait_time}s...")
            time.sleep(wait_time)

            # Reset window after waiting
            _token_tracker["tokens_used"] = 0
            _token_tracker["window_start"] = time.time()
            delay *= 2


def _parse_retry_after(error_msg: str, fallback: int) -> int:
    match = re.search(r"(?:try again in|retry after)\s*([\d.]+)", error_msg, re.IGNORECASE)
    if match:
        return int(float(match.group(1))) + 1
    return fallback