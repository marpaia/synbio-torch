from __future__ import annotations

import pytest

from synbiotorch.config import RunConfig
from synbiotorch.exceptions import ConfigError


def test_fasta_config_roundtrip(tmp_path):
    yaml_text = """
seed: 7
output_dir: runs/test
corpus:
  source: fasta
  path: data/seqs.fasta
  label_key: measure
tokenizer:
  kind: kmer
  k: 4
task:
  kind: supervised
  objective: regression
"""
    path = tmp_path / "run.yaml"
    path.write_text(yaml_text)
    config = RunConfig.from_yaml(path)
    assert config.seed == 7
    assert config.corpus.source == "fasta"
    assert config.tokenizer.k == 4
    # Round-trips through YAML serialization.
    assert "seed: 7" in config.to_yaml()


def test_sbol_db_requires_base_url():
    with pytest.raises(ConfigError):
        RunConfig.model_validate({"corpus": {"source": "sbol_db"}})


def test_classification_requires_num_classes():
    with pytest.raises(ConfigError):
        RunConfig.model_validate(
            {
                "corpus": {"source": "fasta", "path": "x.fasta"},
                "task": {"kind": "supervised", "objective": "classification"},
            }
        )


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ConfigError):
        RunConfig.model_validate(
            {
                "corpus": {"source": "fasta", "path": "x.fasta"},
                "splits": {"ratios": [0.5, 0.4, 0.4]},
            }
        )
