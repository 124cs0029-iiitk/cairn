"""Model calls that wait and retry on rate limits and short network or server problems."""
import time

from .llm import LLMError, chat

TRANSIENT = ("rate", "429", "too many", "tokens per", "getaddrinfo", "connect", "timed out", "timeout",
             "502", "503", "504", "overloaded")


def patient_chat(role, messages, **kwargs):
    for attempt in range(7):
        try:
            return chat(role, messages, **kwargs)
        except LLMError as exc:
            text = str(exc).lower()
            if attempt < 6 and any(word in text for word in TRANSIENT):
                wait = 15 * (attempt + 1)
                print(f"    (temporary problem: {str(exc)[:70]}... waiting {wait}s)")
                time.sleep(wait)
            else:
                raise