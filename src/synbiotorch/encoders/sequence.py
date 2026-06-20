"""Sequence encoder: Design -> tokenized model input."""

from __future__ import annotations

from synbiotorch.encoders.base import EncoderSpec, ModelInput
from synbiotorch.exceptions import ParseError
from synbiotorch.tokenize.base import Tokenizer
from synbiotorch.types import Design


class SequenceEncoder:
    """Encodes the object's raw sequence elements via the configured tokenizer."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, obj: Design) -> ModelInput:
        if obj.sequence is None or not obj.sequence.elements:
            raise ParseError(f"object {obj.iri} has no sequence to encode")
        enc = self.tokenizer.encode(obj.sequence.elements)
        return ModelInput(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            label=obj.label,
        )

    @property
    def output_spec(self) -> EncoderSpec:
        return EncoderSpec(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            max_length=self.tokenizer.max_length,
        )
