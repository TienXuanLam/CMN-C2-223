"""Filesystem containment policy for local knowledge-base files."""

from pathlib import Path


def resolve_project_file(configured_path: str, project_root: Path) -> Path:
    """Resolve a configured file and reject paths outside the project root."""
    candidate = Path(configured_path)
    resolved = (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("configured path escapes project root") from exc
    return resolved
