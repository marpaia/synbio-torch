"""Tests for scripts/normalize_sbol.sh and the committed example input.

The script shells out to the external `sbol` CLI, which is not available in CI, so
these tests inject a stub binary (resolved through the same SBOL_BIN / .env path
as the real one) and assert the script's resolution and format dispatch — then
parse the stub's output through LocalFileCorpus to close the loop.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from sboltorch.data.local import LocalFileCorpus

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "normalize_sbol.sh"
DEMO_GB = REPO_ROOT / "examples" / "data" / "demo_tu.gb"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script; POSIX only")

# A stub `sbol`: ignore the subcommand, find `-o <dest>`, write a tiny SBOL3 doc.
_STUB = """\
#!/usr/bin/env bash
dest=""
while [ $# -gt 0 ]; do
  case "$1" in -o) dest="$2"; shift 2;; *) shift;; esac
done
cat > "$dest" <<'TTL'
@prefix sbol: <http://sbols.org/v3#> .
<https://stub/c1> a sbol:Component ;
    sbol:hasSequence <https://stub/c1/seq> ;
    sbol:hasFeature <https://stub/c1/f1> .
<https://stub/c1/seq> a sbol:Sequence ; sbol:elements "acgtacgt" .
<https://stub/c1/f1> a sbol:SequenceFeature ;
    sbol:role <https://identifiers.org/SO:0000167> ;
    sbol:hasLocation <https://stub/c1/f1/r> .
<https://stub/c1/f1/r> a sbol:Range ; sbol:start 1 ; sbol:end 4 .
TTL
"""


def _make_stub(path: Path) -> Path:
    path.write_text(_STUB)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_repo(tmp_path: Path, env_line: str | None) -> tuple[Path, Path]:
    """A throwaway repo containing the real script and an optional .env line."""
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "normalize_sbol.sh"
    shutil.copy(SCRIPT, script)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    if env_line is not None:
        (tmp_path / ".env").write_text(env_line + "\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(DEMO_GB, raw / "demo_tu.gb")
    return script, raw


def _run(script: Path, raw: Path, out: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(script), str(raw), str(out)],
        env={**env, "NAMESPACE": "https://test/ns"},
        capture_output=True,
        text=True,
    )


def test_demo_input_is_present_and_well_formed():
    text = DEMO_GB.read_text()
    assert text.startswith("LOCUS")
    assert "FEATURES" in text and "ORIGIN" in text and text.rstrip().endswith("//")
    # Four annotated parts: promoter, RBS, CDS, terminator.
    for part in ("promoter", "RBS", "CDS", "terminator"):
        assert part in text


def test_script_resolves_sbol_bin_from_env_file(tmp_path):
    stub = _make_stub(tmp_path / "sbol_stub")
    script, raw = _fake_repo(tmp_path, env_line=f"SBOL_BIN={stub}")
    out = tmp_path / "out"

    # SBOL_BIN absent from the environment -> must be read from .env.
    env = {k: v for k, v in os.environ.items() if k != "SBOL_BIN"}
    result = _run(script, raw, out, env)

    assert result.returncode == 0, result.stderr
    produced = list(out.glob("*.ttl"))
    assert produced, result.stderr
    # The stub's SBOL3 output round-trips through the Component-centric parser.
    objs = list(LocalFileCorpus(produced[0], fmt="sbol"))
    comp = next(o for o in objs if o.sbol_class.endswith("Component"))
    assert comp.sequence and comp.features


def test_explicit_sbol_bin_overrides_env_file(tmp_path):
    real = _make_stub(tmp_path / "real_stub")
    # .env points at a path that does not exist; the explicit env var must win.
    script, raw = _fake_repo(tmp_path, env_line=f"SBOL_BIN={tmp_path / 'missing'}")
    out = tmp_path / "out"

    result = _run(script, raw, out, {**os.environ, "SBOL_BIN": str(real)})
    assert result.returncode == 0, result.stderr
    assert list(out.glob("*.ttl"))


def test_expands_leading_tilde_in_sbol_bin(tmp_path, monkeypatch):
    # A ~-relative SBOL_BIN (as stored in .env) is expanded before invocation.
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    _make_stub(home / "bin" / "sbol_stub")
    monkeypatch.setenv("HOME", str(home))

    script, raw = _fake_repo(tmp_path, env_line="SBOL_BIN=~/bin/sbol_stub")
    out = tmp_path / "out"
    env = {k: v for k, v in os.environ.items() if k != "SBOL_BIN"}
    env["HOME"] = str(home)

    result = _run(script, raw, out, env)
    assert result.returncode == 0, result.stderr
    assert list(out.glob("*.ttl"))
