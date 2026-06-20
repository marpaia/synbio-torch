"""Structure-aware encoder: sequence tokens + injected feature-boundary markers.

The sequence is tokenized with the base tokenizer, but each annotated feature
span is wrapped with role-keyed boundary tokens — e.g. ``[promoter] … [/promoter]``
— and reverse-complement features get an orientation marker. The boundary tokens
are appended to the base vocabulary, so a model embedding sized to
``output_spec.vocab_size`` (a from-scratch encoder, or a pretrained one whose
embeddings are resized) sees the SBOL structure inline with the sequence.
"""

from __future__ import annotations

from synbiotorch.encoders.base import EncoderSpec, ModelInput
from synbiotorch.tokenize.base import Tokenizer
from synbiotorch.types import Design, local_name

# Roles the synthetic generator emits; used as the default marker vocabulary.
DEFAULT_ROLES = (
    "https://identifiers.org/SO:0000167",  # promoter
    "https://identifiers.org/SO:0000139",  # RBS
    "https://identifiers.org/SO:0000316",  # CDS
    "https://identifiers.org/SO:0000141",  # terminator
)
_REVERSE_COMPLEMENT_SUFFIX = "reverseComplement"


class StructureAwareEncoder:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        roles: tuple[str, ...] = DEFAULT_ROLES,
        mark_orientation: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.mark_orientation = mark_orientation

        # cls/sep ids without extending the protocol: encode the empty string.
        empty = tokenizer.encode("").input_ids
        self._cls, self._sep = empty[0], empty[-1]

        # Boundary markers are assigned ids just past the base vocabulary.
        self._ann: dict[str, int] = {}
        next_id = tokenizer.vocab_size
        for role in roles:
            name = local_name(role)
            self._ann[f"start:{name}"] = next_id
            self._ann[f"end:{name}"] = next_id + 1
            next_id += 2
        for key in ("start:feature", "end:feature"):
            self._ann[key] = next_id
            next_id += 1
        if mark_orientation:
            self._ann["rc"] = next_id
            next_id += 1
        self._vocab_size = next_id

    def _start_id(self, roles: tuple[str, ...]) -> int:
        for role in roles:
            key = f"start:{local_name(role)}"
            if key in self._ann:
                return self._ann[key]
        return self._ann["start:feature"]

    def _end_id(self, roles: tuple[str, ...]) -> int:
        for role in roles:
            key = f"end:{local_name(role)}"
            if key in self._ann:
                return self._ann[key]
        return self._ann["end:feature"]

    def encode(self, obj: Design) -> ModelInput:
        if obj.sequence is None or not obj.sequence.elements:
            raise ValueError(f"object {obj.iri} has no sequence to encode")
        seq = obj.sequence.elements
        tok = self.tokenizer

        body: list[int] = []
        if not obj.features:
            body = tok.tokenize_content(seq)
        else:
            located = sorted(
                (f for f in obj.features if f.locations and f.locations[0].start is not None),
                key=lambda f: f.locations[0].start or 0,
            )
            cursor = 0  # 0-based base position consumed
            for feature in located:
                loc = feature.locations[0]
                start = (loc.start or 1) - 1
                end = loc.end or start
                if start > cursor:
                    body += tok.tokenize_content(seq[cursor:start])
                body.append(self._start_id(feature.roles))
                if self.mark_orientation and (loc.orientation or "").endswith(_REVERSE_COMPLEMENT_SUFFIX):
                    body.append(self._ann["rc"])
                body += tok.tokenize_content(seq[start:end])
                body.append(self._end_id(feature.roles))
                cursor = max(cursor, end)
            if cursor < len(seq):
                body += tok.tokenize_content(seq[cursor:])

        body = body[: tok.max_length - 2]
        ids = [self._cls, *body, self._sep]
        return ModelInput(input_ids=ids, attention_mask=[1] * len(ids), label=obj.label)

    @property
    def output_spec(self) -> EncoderSpec:
        return EncoderSpec(
            vocab_size=self._vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            max_length=self.tokenizer.max_length,
        )
