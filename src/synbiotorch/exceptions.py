"""Typed exceptions for synbiotorch.

Parsing and I/O failures raise these explicitly rather than being swallowed, so
callers can distinguish a malformed record from a transport error.
"""

from __future__ import annotations


class SbolmlError(Exception):
    """Base class for all synbiotorch errors."""


class CorpusError(SbolmlError):
    """A corpus could not produce records (bad source, transport, or parse)."""


class ParseError(CorpusError):
    """A source document or record could not be parsed into an Design."""


class SbolDbError(CorpusError):
    """The sbol-db service returned an error or an unexpected payload."""


class ConfigError(SbolmlError):
    """A run configuration is invalid or incomplete."""
