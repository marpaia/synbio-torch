"""GenBank corpus source.

GenBank flat files (`.gb`/`.gbk`) are imported to SBOL 3 in-process by the
native sbol-rs binding, then flattened to ``Design`` records carrying sequence,
SO-mapped features, and the composition graph. ``namespace`` roots the resulting
identities (GenBank carries none).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from synbiotorch import _sbol
from synbiotorch.config import CorpusConfig
from synbiotorch.exceptions import ParseError
from synbiotorch.types import Design

from .files import fingerprint_files, list_files
from .sbol import records_to_designs

_GENBANK_EXTENSIONS = {".gb", ".gbk", ".genbank"}


class GenbankCorpus:
    """Reads ``Design`` records from GenBank files via the sbol-rs binding."""

    def __init__(self, path: str | Path, *, namespace: str, label_key: str | None = None) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.label_key = label_key

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "GenbankCorpus":
        assert config.path is not None and config.namespace is not None  # guaranteed by validation
        return cls(config.path, namespace=config.namespace, label_key=config.label_key)

    def _files(self) -> list[Path]:
        return [p for p in list_files(self.path) if p.suffix.lower() in _GENBANK_EXTENSIONS]

    def __iter__(self) -> Iterator[Design]:
        for file in self._files():
            try:
                raw = _sbol.import_genbank(self.namespace, file.read_text())
            except ValueError as exc:
                raise ParseError(f"failed to import GenBank file {file}: {exc}") from exc
            yield from records_to_designs(json.loads(raw), self.label_key)

    def fingerprint(self) -> str:
        return fingerprint_files(self._files(), self.namespace, self.label_key)
