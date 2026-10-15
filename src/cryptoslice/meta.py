"""Create deterministic metadata and meta/data JSON artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptoslice.schemas import Artifact, Meta


def stable_now_iso() -> str:
    """Return an ISO timestamp, optionally pinned by `CRYPTOSLICE_FIXED_TIME`."""

    fixed = os.environ.get("CRYPTOSLICE_FIXED_TIME")
    if fixed:
        return fixed
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_commit() -> str:
    """Return the short git commit hash, or `nogit` when unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "nogit"
    return result.stdout.strip() or "nogit"


def hash_files(paths: list[Path]) -> str:
    """Return a deterministic SHA-256 over path names and contents."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item).replace("\\", "/")):
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        if path.exists() and path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def make_meta(
    generator: str,
    input_files: list[Path],
    config_files: list[Path] | None = None,
) -> Meta:
    """Build the required metadata block for generated JSON artifacts."""

    return Meta(
        generated_at=stable_now_iso(),
        generator=generator,
        git_commit=git_commit(),
        config_hash=hash_files(config_files or []),
        input_files=[str(path).replace("\\", "/") for path in input_files],
    )


def stable_json(data: Any) -> str:
    """Serialize JSON with stable key ordering and trailing newline."""

    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_artifact(path: Path, meta: Meta, data: dict[str, Any]) -> None:
    """Validate and write a meta/data JSON artifact."""

    artifact = Artifact(meta=meta, data=data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(artifact.model_dump(mode="json")), encoding="utf-8")
