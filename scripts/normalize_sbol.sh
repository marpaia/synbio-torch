#!/usr/bin/env bash
#
# Normalize a directory of mixed biological-design files into native SBOL 3
# Turtle, ready for `source: local` (fmt: sbol).
#
#   GenBank (.gb/.gbk)  -> sbol import-genbank
#   SBOL 2 RDF (.xml/.rdf/.ttl/.nt) -> sbol upgrade
#   SBOL 3 (.ttl/.nt/.rdf/.jsonld)  -> sbol convert (re-serialized to Turtle)
#
# FASTA is intentionally NOT handled here: `sbol import-fasta` writes a header's
# `key=value` into sbol:description rather than a numeric predicate, so labels
# would not survive. Feed labeled FASTA directly via `source: local` (fmt: fasta),
# whose header parsing reads `measure=...` correctly.
#
# Install the `sbol` CLI with: cargo install sbol-cli
#
# Usage:
#   NAMESPACE=https://example.org/mydesigns scripts/normalize_sbol.sh raw/ out/
#
# The `sbol` binary is located in this order: the SBOL_BIN environment variable,
# then an `SBOL_BIN=...` line in the repo-root .env, then `sbol` on PATH. So with
# SBOL_BIN in .env you can just run the script; override per-invocation with
# `SBOL_BIN=/path/to/sbol scripts/normalize_sbol.sh ...`.
#
# GenBank and FASTA carry no namespace; NAMESPACE roots the SBOL 3 top-levels and
# is required when any .gb input is present.

set -euo pipefail

# Resolve the CLI: explicit SBOL_BIN wins, else read it from the repo-root .env
# (the same file that holds WANDB_API_KEY), else fall back to `sbol` on PATH.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${SBOL_BIN:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  SBOL_BIN="$(sed -n 's/^[[:space:]]*SBOL_BIN[[:space:]]*=[[:space:]]*//p' "$REPO_ROOT/.env" | tail -n1)"
  SBOL_BIN="${SBOL_BIN%\"}"; SBOL_BIN="${SBOL_BIN#\"}"   # strip surrounding double quotes
  SBOL_BIN="${SBOL_BIN%\'}"; SBOL_BIN="${SBOL_BIN#\'}"   # strip surrounding single quotes
fi
SBOL="${SBOL_BIN:-sbol}"
SBOL="${SBOL/#\~/$HOME}"   # expand a leading ~ regardless of where SBOL_BIN came from
SRC="${1:?usage: normalize_sbol.sh <src-dir> <out-dir>}"
OUT="${2:?usage: normalize_sbol.sh <src-dir> <out-dir>}"
NAMESPACE="${NAMESPACE:-}"

command -v "$SBOL" >/dev/null 2>&1 || {
  echo "sbol CLI not found: $SBOL (install with 'cargo install sbol-cli', or set SBOL_BIN)" >&2
  exit 1
}
mkdir -p "$OUT"

shopt -s nullglob nocaseglob
count=0

for f in "$SRC"/*; do
  [ -f "$f" ] || continue
  base="$(basename "${f%.*}")"
  dest="$OUT/$base.ttl"
  ext="$(printf '%s' "${f##*.}" | tr '[:upper:]' '[:lower:]')"

  case "$ext" in
    gb|gbk)
      [ -n "$NAMESPACE" ] || { echo "NAMESPACE required to import GenBank: $f" >&2; exit 1; }
      "$SBOL" import-genbank "$f" --namespace "$NAMESPACE" --to turtle -o "$dest"
      ;;
    xml|rdf|nt|ttl|jsonld)
      # Probe for SBOL 3: upgrade SBOL 2, convert SBOL 3 in place.
      if grep -q "sbols.org/v3#" "$f" 2>/dev/null; then
        "$SBOL" convert "$f" --to turtle -o "$dest"
      else
        "$SBOL" upgrade "$f" ${NAMESPACE:+--namespace "$NAMESPACE"} --to turtle -o "$dest"
      fi
      ;;
    fa|fasta|fna|faa)
      echo "skipping FASTA (feed directly as fmt: fasta): $f" >&2
      continue
      ;;
    *)
      echo "skipping unsupported extension: $f" >&2
      continue
      ;;
  esac
  count=$((count + 1))
  echo "normalized: $f -> $dest"
done

echo "done: $count file(s) -> $OUT"
