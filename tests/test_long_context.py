"""Stage-4 long context + modern attention: RoPE architectures, SDPA, and the
long-sequence behavior that distinguishes them from absolute-position models.

The discipline: the attention implementation and rotary base are shown to land on
the built model; a RoPE decoder is shown to actually learn; and the long-context
claim is made concrete — an absolute-position model is capped at its trained
length while a RoPE one runs well past it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from synbiotorch.config import ArchConfig, ModelConfig, RunConfig, TaskConfig, TrainConfig
from synbiotorch.datasets.causal_collator import CausalCollator
from synbiotorch.datasets.packing import PackedDataset
from synbiotorch.engine.trainer import Callback, Trainer
from synbiotorch.models import build_model
from synbiotorch.reproducibility import set_seed
from synbiotorch.tasks.causal import CausalLMTask
from synbiotorch.tokenize.kmer import KmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence

CPU = torch.device("cpu")
EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "configs"


def _causal(model_type: str, *, max_positions: int, hidden: int = 64, vocab: int, pad: int, **arch_kw):
    return build_model(
        ModelConfig(
            from_scratch=True,
            hidden_size=hidden,
            arch=ArchConfig(
                model_type=model_type,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=128,
                max_position_embeddings=max_positions,
                **arch_kw,
            ),
        ),
        TaskConfig(kind="causal"),
        vocab_size=vocab,
        pad_token_id=pad,
    )


def _forward(model: torch.nn.Module, length: int) -> torch.Tensor:
    ids = torch.randint(5, 20, (1, length), dtype=torch.long)
    return model(ids, torch.ones_like(ids))


def test_attn_implementation_is_applied():
    model = _causal("gpt_neox", max_positions=128, vocab=64, pad=0)  # default sdpa
    assert model.lm.config._attn_implementation == "sdpa"
    eager = _causal("gpt_neox", max_positions=128, vocab=64, pad=0, attn_implementation="eager")
    assert eager.lm.config._attn_implementation == "eager"


def test_rope_theta_is_applied():
    model = _causal("llama", max_positions=128, vocab=64, pad=0, rope_theta=12345.0)
    assert float(model.lm.config.rope_theta) == 12345.0


def test_rope_model_runs_past_absolute_limit():
    # An absolute-position model is hard-capped at its trained length...
    gpt2 = _causal("gpt2", max_positions=64, vocab=64, pad=0)
    with pytest.raises(Exception):
        _forward(gpt2, 128)
    # ...while a RoPE model extrapolates to longer sequences without a position table.
    neox = _causal("gpt_neox", max_positions=64, vocab=64, pad=0)
    out = _forward(neox, 128)
    assert out.shape[:2] == (1, 128)


class _History(Callback):
    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        self.rows.append(dict(metrics))


def test_rope_decoder_learns_next_token():
    set_seed(0)
    tok = KmerTokenizer(k=3, max_length=512)
    objs = [
        Design(
            iri=f"https://ex/c{i}",
            record_class="http://sbols.org/v3#Sequence",
            sequence=Sequence(elements=("ACGT" * 30) if i % 2 else ("GGCC" * 30), alphabet=Alphabet.DNA),
        )
        for i in range(240)
    ]
    collator = CausalCollator(tok.pad_token_id)
    train = DataLoader(PackedDataset(objs, tok, block_size=32, which="train"), batch_size=8, collate_fn=collator)
    val = DataLoader(PackedDataset(objs, tok, block_size=32, which="val"), batch_size=8, collate_fn=collator)
    model = _causal("gpt_neox", max_positions=64, vocab=tok.vocab_size, pad=tok.pad_token_id)
    history = _History()
    cfg = TrainConfig(batch_size=8, lr=5e-3, amp=False, max_steps=80, eval_every_n_steps=20)
    Trainer(model, CausalLMTask(), cfg, callbacks=[history], device=CPU).fit(train, val)
    assert history.rows[-1]["val_loss"] < history.rows[0]["val_loss"]


def test_modernbert_mlm_builds_and_runs():
    tok = KmerTokenizer(k=3, max_length=512)
    model = build_model(
        ModelConfig(
            from_scratch=True,
            hidden_size=64,
            arch=ArchConfig(
                model_type="modernbert",
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=128,
                max_position_embeddings=512,
            ),
        ),
        TaskConfig(kind="mlm"),
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
    )
    assert model.lm.config._attn_implementation == "sdpa"
    ids = torch.randint(5, 20, (1, 96), dtype=torch.long)
    logits = model(ids, torch.ones_like(ids))
    assert logits.shape == (1, 96, tok.vocab_size)


def test_long_context_example_config_validates():
    config = RunConfig.from_yaml(EXAMPLES / "pretrain_causal_long.yaml")
    assert config.model.arch.model_type == "gpt_neox"
    assert config.model.arch.attn_implementation == "sdpa"
    assert config.packing.enabled and config.streaming
