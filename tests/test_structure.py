"""Structure-aware encoder: boundary markers, materialization, end-to-end training."""

from __future__ import annotations

import numpy as np

from synbiotorch.config import RunConfig
from synbiotorch.data.materialize import materialize
from synbiotorch.encoders.structure import DEFAULT_ROLES, StructureAwareEncoder
from synbiotorch.pipeline import run_training
from synbiotorch.sources.synthetic import SyntheticCorpus, generate_components
from synbiotorch.tokenize.kmer import KmerTokenizer


def test_encoder_vocab_extends_base_with_markers():
    tok = KmerTokenizer(k=3, max_length=256)
    enc = StructureAwareEncoder(tok, roles=DEFAULT_ROLES, mark_orientation=True)
    # 2 markers per role + generic start/end + one rc marker.
    assert enc.output_spec.vocab_size == tok.vocab_size + 2 * len(DEFAULT_ROLES) + 2 + 1


def test_encoder_injects_boundary_markers():
    tok = KmerTokenizer(k=3, max_length=256)
    enc = StructureAwareEncoder(tok)
    comp = generate_components(1, seed=1)[0]
    out = enc.encode(comp)
    marker_ids = [i for i in out.input_ids if i >= tok.vocab_size]
    # Four features -> at least four start + four end markers present.
    assert len(marker_ids) >= 8
    assert len(out.input_ids) == len(out.attention_mask)


def test_encoder_marks_reverse_complement_orientation():
    tok = KmerTokenizer(k=3, max_length=256)
    enc = StructureAwareEncoder(tok)
    rc_id = enc._ann["rc"]
    # Across several components at least one feature is reverse-complemented.
    seen_rc = any(rc_id in enc.encode(c).input_ids for c in generate_components(20, seed=2))
    assert seen_rc


def test_encoder_without_features_falls_back_to_plain_sequence():
    from synbiotorch.types import Alphabet, Design, Sequence

    tok = KmerTokenizer(k=3, max_length=256)
    enc = StructureAwareEncoder(tok)
    obj = Design(iri="x", record_class="c", sequence=Sequence(elements="ACGTACGTACGT", alphabet=Alphabet.DNA))
    out = enc.encode(obj)
    assert all(i < tok.vocab_size for i in out.input_ids)  # no markers


def test_materialize_preserves_features_and_graph(tmp_path):
    corpus = SyntheticCorpus(6, seed=4)
    mat = materialize(corpus, tmp_path)
    objs = mat.read_all()
    assert all(len(o.features) == 4 for o in objs)
    assert all(o.neighbors is not None for o in objs)
    # Locations survive the round-trip.
    assert objs[0].features[0].locations[0].start == 1


def test_structure_aware_end_to_end(tmp_path):
    config = RunConfig.model_validate(
        {
            "seed": 1,
            "output_dir": str(tmp_path / "run"),
            "corpus": {"source": "synthetic", "n": 48, "label_key": "strength", "cache_dir": str(tmp_path / "cache")},
            "tokenizer": {"kind": "kmer", "k": 3, "max_length": 256},
            "encoder": {"kind": "structure_aware"},
            "model": {
                "from_scratch": True,
                "hidden_size": 48,
                "arch": {
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 96,
                    "max_position_embeddings": 256,
                },
            },
            "task": {"kind": "supervised", "objective": "regression"},
            "splits": {"strategy": "random", "ratios": [0.7, 0.15, 0.15]},
            "train": {"batch_size": 8, "epochs": 2, "lr": 5.0e-3, "amp": False},
        }
    )
    metrics = run_training(config)
    assert "val_mae" in metrics and np.isfinite(metrics["val_mae"])
