"""Shared helpers for file-backed corpus sources."""

from __future__ import annotations

import hashlib
from pathlib import Path


def list_files(path: str | Path) -> list[Path]:
    """Return the file (or every file under a directory), sorted for determinism."""
    root = Path(path)
    if root.is_dir():
        return sorted(p for p in root.rglob("*") if p.is_file())
    return [root]


def fingerprint_files(paths: list[Path], *extra: object) -> str:
    """A stable content hash over file identity/size/mtime plus extra parameters."""
    h = hashlib.sha256()
    for file in paths:
        stat = file.stat()
        h.update(str(file).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    for item in extra:
        h.update(repr(item).encode())
    return h.hexdigest()[:16]
