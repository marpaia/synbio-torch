"""Tabular corpus source for labeled sequence datasets.

Reads CSV/TSV where one column holds the sequence and (optionally) another holds
a numeric label — the shape most public sequence-activity/expression/fitness
datasets ship in. One ``Design`` per row.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from synbiotorch.config import CorpusConfig
from synbiotorch.exceptions import ParseError
from synbiotorch.types import Alphabet, Design, Sequence

from .fasta import detect_alphabet
from .files import fingerprint_files, list_files

SBOL3 = "http://sbols.org/v3#"
_TABLE_EXTENSIONS = {".csv", ".tsv"}


class TableCorpus:
    """Reads ``Design`` records from CSV/TSV files (one row per record)."""

    def __init__(
        self,
        path: str | Path,
        *,
        sequence_column: str,
        label_column: str | None = None,
        id_column: str | None = None,
        alphabet: str = "auto",
    ) -> None:
        self.path = Path(path)
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.id_column = id_column
        self.alphabet = alphabet

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "TableCorpus":
        assert config.path is not None and config.sequence_column is not None  # guaranteed by validation
        return cls(
            config.path,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            id_column=config.id_column,
            alphabet=config.alphabet,
        )

    def _files(self) -> list[Path]:
        return [p for p in list_files(self.path) if p.suffix.lower() in _TABLE_EXTENSIONS]

    def _alphabet_for(self, elements: str) -> Alphabet:
        if self.alphabet != "auto":
            return Alphabet(self.alphabet.upper())
        return detect_alphabet(elements)

    def __iter__(self) -> Iterator[Design]:
        for file in self._files():
            delimiter = "\t" if file.suffix.lower() == ".tsv" else ","
            with file.open(newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if self.sequence_column not in (reader.fieldnames or []):
                    raise ParseError(f"{file}: missing sequence column {self.sequence_column!r}")
                for index, row in enumerate(reader):
                    elements = (row.get(self.sequence_column) or "").strip()
                    if not elements:
                        continue
                    iri = (row.get(self.id_column) if self.id_column else None) or f"{file.stem}:{index}"
                    yield Design(
                        iri=iri,
                        record_class=f"{SBOL3}Sequence",
                        display_id=iri,
                        sequence=Sequence(elements=elements, alphabet=self._alphabet_for(elements)),
                        label=_parse_label(row.get(self.label_column)) if self.label_column else None,
                        raw=dict(row),
                    )

    def fingerprint(self) -> str:
        return fingerprint_files(self._files(), self.sequence_column, self.label_column, self.id_column, self.alphabet)


def _parse_label(value: str | None) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    return int(num) if num.is_integer() else num
