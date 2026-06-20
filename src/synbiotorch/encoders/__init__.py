"""Encoders: the input-modality plug point."""

from __future__ import annotations

from synbiotorch.encoders.base import Encoder, EncoderSpec, ModelInput, build_encoder
from synbiotorch.encoders.graph import GraphEncoder, GraphSpec
from synbiotorch.encoders.sequence import SequenceEncoder
from synbiotorch.encoders.structure import StructureAwareEncoder

__all__ = [
    "Encoder",
    "EncoderSpec",
    "ModelInput",
    "build_encoder",
    "SequenceEncoder",
    "StructureAwareEncoder",
    "GraphEncoder",
    "GraphSpec",
]
