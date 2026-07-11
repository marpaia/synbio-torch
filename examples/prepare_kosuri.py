"""Prepare the Kosuri et al. 2013 promoter x RBS composability dataset.

Kosuri et al., "Composability of regulatory sequences controlling transcription
and translation in Escherichia coli," PNAS 110:14024 (2013),
doi:10.1073/pnas.1301301110. The library pairs 112 promoters with 111 RBSs and
measures protein and RNA levels for every combination, so each construct is a
genuine composition of two annotated parts reused across the library.

Each usable construct becomes an SBOL 3 ``Component``: the concatenated
promoter+RBS sequence, a promoter ``SubComponent`` and an RBS ``SubComponent``
with Sequence-Ontology roles and Range locations, and each pointing at the part
it instantiates (part identities are shared across constructs, so the
composition graphs carry real structure). Three annotations ride on each
construct: ``prot`` (the log10 protein-level label) and two split assignments,
``split_random`` and ``split_partout``, selected by a run's ``splits.column``.

The three PNAS Supporting Information files are not open-access, so they are not
redistributed here. Download them once from the PMC mirror into ``data/kosuri/``:

    https://pmc.ncbi.nlm.nih.gov/articles/PMC3752251/  (Supplementary Materials)
      1301301110_sd01.xls -> data/kosuri/sd01.xls   (promoters)
      1301301110_sd02.xls -> data/kosuri/sd02.xls   (RBSs)
      1301301110_sd03.xls -> data/kosuri/sd03.xls   (constructs)

Then run (pandas + xlrd read the legacy .xls; both are prep-only):

    env -u VIRTUAL_ENV uv run --with pandas --with xlrd \
        python examples/prepare_kosuri.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kosuri"
SBOL_DIR = DATA_DIR / "sbol"
MANIFEST = DATA_DIR / "kosuri_manifest.json"
INPUTS = {
    "sd01": DATA_DIR / "sd01.xls",
    "sd02": DATA_DIR / "sd02.xls",
    "sd03": DATA_DIR / "sd03.xls",
}

NS = "https://synbiotorch.org/kosuri/"
SBOL3 = "http://sbols.org/v3#"
XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
ORIENTATION_INLINE = f"{SBOL3}inline"
ROLE = {
    "promoter": "https://identifiers.org/SO:0000167",
    "rbs": "https://identifiers.org/SO:0000139",
}

VAL_FRACTION = 0.10
PARTOUT_HOLDOUT_FRACTION = 0.10
SEED = 42
SHARD_SIZE = 4000
# Guard against a silently-changed upstream: the joined, quality-filtered library
# must match the partition sizes this script was written against.
EXPECTED_CONSTRUCTS = 11_696

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _require_inputs() -> None:
    missing = [str(p) for p in INPUTS.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "missing Kosuri SI files:\n  "
            + "\n  ".join(missing)
            + "\n\nDownload them from https://pmc.ncbi.nlm.nih.gov/articles/PMC3752251/"
            " (Supplementary Materials) into data/kosuri/ and re-run. See this file's"
            " docstring for the exact file mapping."
        )


def _clean(value: object) -> str:
    return str(value).strip().strip('"').strip()


def _sanitize(value: str) -> str:
    return _SAFE.sub("_", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load() -> pd.DataFrame:
    prom = pd.read_excel(INPUTS["sd01"], sheet_name="Promoters", engine="xlrd")
    rbs = pd.read_excel(INPUTS["sd02"], sheet_name="RBSs", engine="xlrd")
    con = pd.read_excel(INPUTS["sd03"], sheet_name="Constructs", engine="xlrd")

    prom_seq = {_clean(r.Promoter): _clean(r.Sequence).replace(" ", "").upper() for r in prom.itertuples()}
    rbs_seq = {_clean(r.RBS): _clean(r.Sequence).replace(" ", "").upper() for r in rbs.itertuples()}

    prot = pd.to_numeric(con["prot"], errors="coerce")
    # Drop constructs the authors flagged as bad protein measurements, or with a
    # missing/non-positive level (log10 needs prot > 0).
    bad = con["bad.prot"].fillna(False).astype(bool)
    keep = (~bad) & prot.notna() & (prot > 0)

    df = pd.DataFrame(
        {
            "promoter_id": con.loc[keep, "Promoter"].map(_clean).values,
            "rbs_id": con.loc[keep, "RBS"].map(_clean).values,
            "log_prot": np.log10(prot[keep].astype(float)).values,
        }
    )
    df["pseq"] = df["promoter_id"].map(prom_seq)
    df["rseq"] = df["rbs_id"].map(rbs_seq)
    # Drop constructs whose parts carry no listed sequence in sd01/sd02.
    df = df.dropna(subset=["pseq", "rseq"]).reset_index(drop=True)
    df["construct_id"] = df["promoter_id"].map(_sanitize) + "__" + df["rbs_id"].map(_sanitize)
    df = df.drop_duplicates(subset="construct_id").reset_index(drop=True)
    return df


def _random_split(ids: list[str], seed: int) -> dict[str, str]:
    order = list(ids)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_test = int(round(0.10 * n))
    n_val = int(round(VAL_FRACTION * n))
    assign: dict[str, str] = {}
    for i, cid in enumerate(order):
        if i < n_test:
            assign[cid] = "test"
        elif i < n_test + n_val:
            assign[cid] = "val"
        else:
            assign[cid] = "train"
    return assign


def _partout_split(df: pd.DataFrame, seed: int, holdout: float) -> dict[str, str]:
    """Hold out whole parts: test constructs use a promoter or RBS never seen in
    training, so the split measures generalization to novel part combinations."""
    rng = random.Random(seed)
    prom_ids = sorted(df["promoter_id"].unique())
    rbs_ids = sorted(df["rbs_id"].unique())
    held_prom = set(rng.sample(prom_ids, max(1, int(round(holdout * len(prom_ids))))))
    held_rbs = set(rng.sample(rbs_ids, max(1, int(round(holdout * len(rbs_ids))))))

    assign: dict[str, str] = {}
    seen: list[str] = []
    for row in df.itertuples():
        if row.promoter_id in held_prom or row.rbs_id in held_rbs:
            assign[row.construct_id] = "test"
        else:
            seen.append(row.construct_id)
    rng.shuffle(seen)
    n_val = int(round(VAL_FRACTION * len(seen)))
    for i, cid in enumerate(seen):
        assign[cid] = "val" if i < n_val else "train"
    return assign


def _emit_sbol(df: pd.DataFrame, rand: dict[str, str], part: dict[str, str]) -> int:
    if SBOL_DIR.exists():
        for old in SBOL_DIR.glob("*.ttl"):
            old.unlink()
    SBOL_DIR.mkdir(parents=True, exist_ok=True)

    def iri(value: str) -> str:
        return f"<{value}>"

    def lit(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def integer(value: int) -> str:
        return f'"{value}"^^<{XSD_INTEGER}>'

    shards = 0
    for shard_start in range(0, len(df), SHARD_SIZE):
        chunk = df.iloc[shard_start : shard_start + SHARD_SIZE]
        lines: list[str] = []
        for row in chunk.itertuples():
            comp = f"{NS}construct/{row.construct_id}"
            seq_iri = f"{comp}/sequence"
            elements = row.pseq + row.rseq
            lines.append(f"{iri(comp)} {RDF_TYPE} {iri(f'{SBOL3}Component')} .")
            lines.append(f"{iri(comp)} {iri(f'{SBOL3}hasSequence')} {iri(seq_iri)} .")
            lines.append(f"{iri(seq_iri)} {RDF_TYPE} {iri(f'{SBOL3}Sequence')} .")
            lines.append(f"{iri(seq_iri)} {iri(f'{SBOL3}elements')} {lit(elements)} .")

            spans = (
                ("promoter", row.promoter_id, 1, len(row.pseq)),
                ("rbs", row.rbs_id, len(row.pseq) + 1, len(row.pseq) + len(row.rseq)),
            )
            for role, part_id, start, end in spans:
                feat = f"{comp}/{role}"
                part_iri = f"{NS}part/{role}/{_sanitize(part_id)}"
                loc = f"{feat}/loc0"
                lines.append(f"{iri(comp)} {iri(f'{SBOL3}hasFeature')} {iri(feat)} .")
                lines.append(f"{iri(feat)} {RDF_TYPE} {iri(f'{SBOL3}SubComponent')} .")
                lines.append(f"{iri(feat)} {iri(f'{SBOL3}instanceOf')} {iri(part_iri)} .")
                lines.append(f"{iri(feat)} {iri(f'{SBOL3}role')} {iri(ROLE[role])} .")
                lines.append(f"{iri(feat)} {iri(f'{SBOL3}hasLocation')} {iri(loc)} .")
                lines.append(f"{iri(loc)} {RDF_TYPE} {iri(f'{SBOL3}Range')} .")
                lines.append(f"{iri(loc)} {iri(f'{SBOL3}start')} {integer(start)} .")
                lines.append(f"{iri(loc)} {iri(f'{SBOL3}end')} {integer(end)} .")
                lines.append(f"{iri(loc)} {iri(f'{SBOL3}orientation')} {iri(ORIENTATION_INLINE)} .")

            lines.append(f"{iri(comp)} {iri(f'{NS}prot')} {lit(f'{row.log_prot:.6f}')} .")
            lines.append(f"{iri(comp)} {iri(f'{NS}split_random')} {lit(rand[row.construct_id])} .")
            lines.append(f"{iri(comp)} {iri(f'{NS}split_partout')} {lit(part[row.construct_id])} .")

        (SBOL_DIR / f"part-{shards:05d}.ttl").write_text("\n".join(lines) + "\n")
        shards += 1
    return shards


def _counts(assign: dict[str, str]) -> dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for value in assign.values():
        out[value] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--holdout", type=float, default=PARTOUT_HOLDOUT_FRACTION)
    args = parser.parse_args()

    _require_inputs()
    df = _load()
    if len(df) != EXPECTED_CONSTRUCTS:
        print(f"warning: {len(df)} usable constructs (expected {EXPECTED_CONSTRUCTS}); upstream files may have changed")

    ids = df["construct_id"].tolist()
    rand = _random_split(ids, args.seed)
    part = _partout_split(df, args.seed, args.holdout)
    shards = _emit_sbol(df, rand, part)

    manifest = {
        "source": "Kosuri et al. 2013, PNAS 10.1073/pnas.1301301110",
        "n_constructs": len(df),
        "n_promoters": int(df["promoter_id"].nunique()),
        "n_rbs": int(df["rbs_id"].nunique()),
        "label": "log10(prot)",
        "seed": args.seed,
        "partout_holdout_fraction": args.holdout,
        "split_random": _counts(rand),
        "split_partout": _counts(part),
        "sbol_shards": shards,
        "inputs_sha256": {name: _sha256(path) for name, path in INPUTS.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {len(df)} constructs to {shards} SBOL shard(s) in {SBOL_DIR}")
    print(f"  promoters={manifest['n_promoters']}  rbs={manifest['n_rbs']}")
    print(f"  split_random  : {manifest['split_random']}")
    print(f"  split_partout : {manifest['split_partout']}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
