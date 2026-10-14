"""Provider-agnostic LLM access through LiteLLM.

Calls are made by *role* (generator, verifier, judge, vlm) so the generator and
the verifier can sit on different providers. Deterministic calls are cached on
disk, and each session is rate limited so a demo cannot exhaust free quotas.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import defaultdict, deque

from .config import get_settings


class LLMError(RuntimeError):
    pass


class RateLimitExceeded(LLMError):
    pass


class RateLimiter:
    """Sliding-window limiter, keyed by session id."""

    def __init__(self, max_calls: int, window_s: float = 3600.0):
        self.max_calls = max_calls
        self.window_s = window_s
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._calls[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.max_calls:
                raise RateLimitExceeded(
                    f"Limit of {self.max_calls} model calls per hour reached for this session."
                )
            q.append(now)


class _Cache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, response: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?)", (key, response))
            self._conn.commit()


_limiter: RateLimiter | None = None
_cache: _Cache | None = None


def _get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().calls_per_session_per_hour)
    return _limiter


def _get_cache() -> _Cache:
    global _cache
    if _cache is None:
        _cache = _Cache(get_settings().cache_path)
    return _cache


def cache_key(model: str, messages: list[dict], temperature: float, json_mode: bool) -> str:
    blob = json.dumps([model, messages, temperature, json_mode], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chat(
    role: str,
    messages: list[dict],
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    session_id: str = "global",
    use_cache: bool | None = None,
    max_tokens: int = 1500,
) -> str:
    """Return the text of a completion from the model configured for `role`.

    Caching defaults to on only for temperature 0, so sampled generations are
    not frozen into identical outputs.
    """
    model = get_settings().model_for(role)
    cache_on = (temperature == 0.0) if use_cache is None else use_cache
    key = cache_key(model, messages, temperature, json_mode)

    if cache_on:
        hit = _get_cache().get(key)
        if hit is not None:
            return hit

    _get_limiter().check(session_id)

    try:
        import litellm  # imported lazily: keeps startup and tests light
    except ImportError as exc:  # pragma: no cover
        raise LLMError("litellm is not installed") from exc

    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature,
                    "max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = litellm.completion(**kwargs)
    except Exception as exc:
        raise LLMError(f"{role} model call failed ({model}): {exc}") from exc

    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise LLMError(f"{role} model returned an empty response ({model})")
    if cache_on:
        _get_cache().put(key, text)
    return text
