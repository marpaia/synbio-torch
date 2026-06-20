"""FASTA corpus source.

One ``Design`` per record. Labels are parsed from ``key=value`` tokens in the
header when ``label_key`` is set. The alphabet is auto-detected (DNA/RNA/protein)
unless overridden, so protein and RNA FASTA feed the right tokenizers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from synbiotorch.config import CorpusConfig
from synbiotorch.types import Alphabet, Design, Sequence

from .files import fingerprint_files, list_files

SBOL3 = "http://sbols.org/v3#"
_FASTA_EXTENSIONS = {".fa", ".fasta", ".fna", ".faa"}
# Characters that, by themselves, identify a sequence as nucleotide (IUPAC).
_NUCLEOTIDE = set("ACGTUNRYSWKMBDHVacgtunryswkmbdhv")


def detect_alphabet(elements: str) -> Alphabet:
    """Infer DNA/RNA/protein from sequence content."""
    letters = {c for c in elements if c.isalpha()}
    if letters and letters <= _NUCLEOTIDE:
        return Alphabet.RNA if ("U" in letters or "u" in letters) else Alphabet.DNA
    return Alphabet.PROTEIN if letters else Alphabet.DNA


class FastaCorpus:
    """Reads ``Design`` records from a FASTA file or a directory of them."""

    def __init__(self, path: str | Path, *, label_key: str | None = None, alphabet: str = "auto") -> None:
        self.path = Path(path)
        self.label_key = label_key
        self.alphabet = alphabet

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "FastaCorpus":
        assert config.path is not None  # guaranteed by CorpusConfig validation
        return cls(config.path, label_key=config.label_key, alphabet=config.alphabet)

    def _files(self) -> list[Path]:
        return [p for p in list_files(self.path) if p.suffix.lower() in _FASTA_EXTENSIONS]

    def _alphabet_for(self, elements: str) -> Alphabet:
        if self.alphabet != "auto":
            return Alphabet(self.alphabet.upper())
        return detect_alphabet(elements)

    def __iter__(self) -> Iterator[Design]:
        for file in self._files():
            yield from self._parse(file)

    def _parse(self, file: Path) -> Iterator[Design]:
        header: str | None = None
        chunks: list[str] = []

        def flush() -> Design | None:
            if header is None:
                return None
            seq_id = header.split()[0] if header.split() else header
            label = _label_from_header(header, self.label_key) if self.label_key else None
            elements = "".join(chunks)
            return Design(
                iri=seq_id,
                record_class=f"{SBOL3}Sequence",
                display_id=seq_id,
                sequence=Sequence(elements=elements, alphabet=self._alphabet_for(elements)),
                label=label,
                raw={"header": header},
            )

        with file.open() as handle:
            for line in handle:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    obj = flush()
                    if obj is not None:
                        yield obj
                    header = line[1:].strip()
                    chunks = []
                elif line:
                    chunks.append(line.strip())
        obj = flush()
        if obj is not None:
            yield obj

    def fingerprint(self) -> str:
        return fingerprint_files(self._files(), self.label_key, self.alphabet)


def _label_from_header(header: str, label_key: str) -> float | int | None:
    for token in header.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key == label_key:
                try:
                    num = float(value)
                except ValueError:
                    return None
                return int(num) if num.is_integer() else num
    return None
