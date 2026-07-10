"""Prepare a Triton-free local copy of DNABERT-2 for Apple Silicon / CPU.

DNABERT-2's remote modeling code imports Triton for an optional flash-attention
kernel. Triton has no Apple-Silicon build, and Transformers' static import check
refuses to load the model when it cannot import Triton, on any device. The flash
path is already guarded by a try/except that falls back to standard attention, so
the fix is purely to remove the Triton import that the static check trips on.

This downloads DNABERT-2 to a local directory and neutralizes the optional flash
file (and any stray ``import triton``) in place. Point ``model.backbone`` and
``tokenizer.model_name`` at the resulting directory. This keeps the workaround
local to the model rather than shimming the global environment, which would break
Transformers' own Triton probing.

    uv run --with huggingface_hub python examples/prepare_dnabert2.py
"""

from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "zhihan1996/DNABERT-2-117M"
DEST = Path(__file__).resolve().parent.parent / "data" / "dnabert2"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO, local_dir=str(DEST))

    # The flash kernel is imported only inside a try/except in bert_layers.py;
    # replacing it with a no-op symbol keeps that import satisfied while dropping
    # the Triton dependency the static check trips on.
    flash = DEST / "flash_attn_triton.py"
    if flash.exists():
        flash.write_text(
            "# Neutralized for Triton-free loading; DNABERT-2 falls back to\n"
            "# standard attention when this symbol is None.\n"
            "flash_attn_qkvpacked_func = None\n"
        )

    # Belt and suspenders: comment out any remaining top-level triton imports so
    # Transformers' check_imports never requires the package.
    for py in DEST.glob("*.py"):
        text = py.read_text()
        patched = re.sub(r"(?m)^(\s*)(import triton.*|from triton.*)$", r"\1pass  # \2", text)
        if patched != text:
            py.write_text(patched)

    print(f"DNABERT-2 prepared at {DEST}")
    print("Set model.backbone and tokenizer.model_name to this path.")


if __name__ == "__main__":
    main()
