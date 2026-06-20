"""The Encoder protocol — the modality plug point.

Each input modality (sequence, structure-aware sequence, composition graph) is
a different Encoder turning an Design into a ``ModelInput``. The training
engine consumes ModelInput and never knows which modality produced it, which is
what makes "support all three" tractable without forking the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from synbiotorch.types import Design


@dataclass(frozen=True)
class ModelInput:
    """Framework-neutral model input: plain Python, converted to tensors by the collator."""

    input_ids: list[int]
    attention_mask: list[int]
    label: float | int | None = None


@dataclass(frozen=True)
class EncoderSpec:
    """Static description of an encoder's outputs, used to size the model."""

    vocab_size: int
    pad_token_id: int
    mask_token_id: int | None
    max_length: int


@runtime_checkable
class SupportsEncode(Protocol):
    """Minimal contract used by the dataset: turn an object into a model input.

    The return type is modality-specific (``ModelInput`` for tensor encoders, a
    PyG ``Data`` for the graph encoder), so it is intentionally unconstrained.
    """

    def encode(self, obj: Design) -> object: ...


@runtime_checkable
class Encoder(SupportsEncode, Protocol):
    def encode(self, obj: Design) -> ModelInput: ...

    @property
    def output_spec(self) -> EncoderSpec: ...


def build_encoder(encoder_config: "object", tokenizer: "object") -> Encoder:  # noqa: ANN001
    """Construct the encoder named by ``encoder_config.kind``."""
    from ..config import EncoderConfig
    from ..tokenize.base import Tokenizer

    assert isinstance(encoder_config, EncoderConfig)
    assert isinstance(tokenizer, Tokenizer)
    if encoder_config.kind == "sequence":
        from .sequence import SequenceEncoder

        return SequenceEncoder(tokenizer)
    if encoder_config.kind == "structure_aware":
        from .structure import DEFAULT_ROLES, StructureAwareEncoder

        return StructureAwareEncoder(
            tokenizer,
            roles=encoder_config.roles or DEFAULT_ROLES,
            mark_orientation=encoder_config.mark_orientation,
        )
    # The graph encoder produces PyG Data (not ModelInput), so it lives outside
    # this tensor-encoder factory; the pipeline constructs it on the graph path.
    raise NotImplementedError(f"encoder kind '{encoder_config.kind}' is constructed via the graph pipeline path")
