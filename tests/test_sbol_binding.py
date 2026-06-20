"""The native sbol-rs binding (``synbiotorch._sbol``).

Confirms the in-process parsing entry points return structured records for
GenBank, SBOL3, and SBOL2 inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import synbiotorch._sbol as sbol

FIXTURES = Path(__file__).parent / "fixtures" / "sbol"
DEMO_GENBANK = Path(__file__).parent.parent / "examples" / "data" / "demo_tu.gb"


def test_import_genbank_yields_component_with_features() -> None:
    records = json.loads(sbol.import_genbank("https://example.org/demo", DEMO_GENBANK.read_text()))
    assert len(records) == 1
    component = records[0]
    assert component["record_class"].endswith("Component")
    assert len(component["sequence"]["elements"]) == 120
    assert len(component["features"]) == 4
    for feature in component["features"]:
        assert feature["locations"], "each GenBank feature carries a location"
        assert feature["locations"][0]["start"] is not None


def test_read_sbol3_yields_components() -> None:
    text = (FIXTURES / "sbol3" / "toggle_switch.ttl").read_text()
    records = json.loads(sbol.read_sbol3(text, "ttl"))
    assert records
    assert any(r["features"] for r in records), "a composite design exposes features"


def test_upgrade_sbol2_yields_sequence() -> None:
    text = (FIXTURES / "sbol2" / "pICH44179.xml").read_text()
    records = json.loads(sbol.upgrade_sbol2(text, "rdf", "https://example.org/ns"))
    sequence_records = [r for r in records if r["sequence"]]
    assert sequence_records
    assert any(len(r["sequence"]["elements"]) > 0 for r in sequence_records)


def test_unknown_format_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        sbol.read_sbol3("", "bogus")
