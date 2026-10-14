"""Central settings. Everything tunable comes from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROVIDER_KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")

# Provider prefix used in LiteLLM model strings -> the key it needs
_PREFIX_TO_KEY = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    root: Path
    static_dir: Path
    kb_dir: Path
    runtime_dir: Path
    db_path: Path
    cache_path: Path

    generator_model: str
    verifier_model: str
    judge_model: str
    vlm_model: str
    stt_model: str
    embed_model: str
    rerank_model: str

    refusal_threshold: float
    max_upload_mb: int
    max_video_minutes: int
    calls_per_session_per_hour: int

    def model_for(self, role: str) -> str:
        try:
            return getattr(self, f"{role}_model")
        except AttributeError as exc:
            raise ValueError(f"Unknown model role: {role!r}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    runtime = ROOT / "data" / "runtime"
    return Settings(
        root=ROOT,
        static_dir=ROOT / "static",
        kb_dir=ROOT / "data" / "kb",
        runtime_dir=runtime,
        db_path=runtime / "app.db",
        cache_path=runtime / "llm_cache.db",
        generator_model=_env("GENERATOR_MODEL", "gemini/gemini-2.5-flash"),
        verifier_model=_env("VERIFIER_MODEL", "groq/llama-3.3-70b-versatile"),
        judge_model=_env("JUDGE_MODEL", "groq/llama-3.3-70b-versatile"),
        vlm_model=_env("VLM_MODEL", "gemini/gemini-2.5-flash"),
        stt_model=_env("STT_MODEL", "groq/whisper-large-v3"),
        embed_model=_env("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        rerank_model=_env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        refusal_threshold=float(_env("REFUSAL_THRESHOLD", "0.58")),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", "50")),
        max_video_minutes=int(_env("MAX_VIDEO_MINUTES", "20")),
        calls_per_session_per_hour=int(_env("CALLS_PER_SESSION_PER_HOUR", "120")),
    )


def configured_keys() -> dict[str, bool]:
    return {name: bool(os.getenv(name, "").strip()) for name in PROVIDER_KEYS}


def role_status(settings: Settings | None = None) -> dict[str, tuple[str, bool]]:
    """For each model role: (model string, whether its provider key is present)."""
    s = settings or get_settings()
    keys = configured_keys()
    out: dict[str, tuple[str, bool]] = {}
    for role in ("generator", "verifier", "judge", "vlm", "stt"):
        model = s.model_for(role)
        prefix = model.split("/", 1)[0]
        needed = _PREFIX_TO_KEY.get(prefix)
        out[role] = (model, bool(needed and keys.get(needed)))
    return out
