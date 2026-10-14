"""Publish the project to a Hugging Face Space and set its secrets.

Usage (from the project folder):
    python scripts/deploy.py YOUR_USERNAME/cairn --dry-run     # list what would be uploaded
    python scripts/deploy.py YOUR_USERNAME/cairn               # create the Space, set secrets, upload

Needs a token with write access, either in the HF_TOKEN environment variable or saved by `hf auth login`.
Provider keys are read from your local .env and stored as Space secrets; .env itself is never uploaded.
Course files under static/ are uploaded only if a course list in data/course/ refers to them.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
SECRET_KEYS = ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
VARIABLE_KEYS = ["GENERATOR_MODEL", "VERIFIER_MODEL", "JUDGE_MODEL", "VLM_MODEL", "STT_MODEL",
                 "REFUSAL_THRESHOLD"]
IGNORE = [".env", ".venv/*", ".git/*", "*/__pycache__/*", "__pycache__/*", "*.pyc", ".pytest_cache/*",
          "data/runtime/*", "data/raw/*"]


def referenced_static_files(root: Path) -> set[str]:
    """Course files under static/ that a course list in data/course/ actually uses."""
    used = set()
    for manifest in (root / "data" / "course").glob("*.json"):
        for source in json.loads(manifest.read_text(encoding="utf-8")).get("sources", []):
            if source.get("static_path"):
                used.add("static/" + source["static_path"])
    return used


def plan_upload(root: Path) -> list[tuple[str, int]]:
    """(relative path, size in bytes) of every file that would be uploaded."""
    from huggingface_hub.utils import filter_repo_objects

    files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    kept = filter_repo_objects(files, ignore_patterns=IGNORE)
    used = referenced_static_files(root)
    kept = [f for f in kept if not f.startswith("static/courses/") or f in used]   # skip unused course files
    return sorted((f, (root / f).stat().st_size) for f in kept)


def deploy(repo_id: str, root: Path, api, env: dict) -> None:
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    for key in SECRET_KEYS:
        if env.get(key):
            api.add_space_secret(repo_id, key, env[key])
    for key in VARIABLE_KEYS:
        if env.get(key):
            api.add_space_variable(repo_id, key, env[key])
    planned = [name for name, _ in plan_upload(root)]          # exactly what the preview lists
    api.upload_folder(folder_path=str(root), repo_id=repo_id, repo_type="space",
                      allow_patterns=planned, ignore_patterns=IGNORE, commit_message="Deploy Cairn")


def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1 or "/" not in args[0]:
        sys.exit("Usage: python scripts/deploy.py YOUR_USERNAME/cairn [--dry-run]")
    repo_id = args[0]

    plan = plan_upload(ROOT)
    total_mb = sum(size for _, size in plan) / 1e6
    print(f"{len(plan)} files, {total_mb:.1f} MB would be uploaded. Largest:")
    for name, size in sorted(plan, key=lambda x: -x[1])[:5]:
        print(f"  {size / 1e6:6.1f} MB  {name}")
    print(".env uploaded:", any(name == ".env" for name, _ in plan))
    if "--dry-run" in argv:
        return

    from huggingface_hub import HfApi

    deploy(repo_id, ROOT, HfApi(token=os.getenv("HF_TOKEN")), dotenv_values(ROOT / ".env"))
    print(f"\nUploaded. The build starts now (a few minutes): https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main(sys.argv[1:])