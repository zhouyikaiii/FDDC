#!/bin/bash
# Overlay the DDC-specific files onto a pinned Holosoma checkout, with verification.
# Usage: bash apply_ddc_to_holosoma.sh <path-to-holosoma-checkout>
#   - checks the checkout is at the exact upstream commit DDC was built on
#   - checks the DDC files here match their recorded checksums (ddc_src/SHA256SUMS)
#   - copies them into src/holosoma/holosoma/<same relative path>, replacing upstream
set -euo pipefail
PIN=5b61d5768bc8e44710e2983db6263e174193981c   # amazon-far/holosoma, 2026-07-24 (paper Table 6 stack)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/ddc_src"
HOLO="${1:?usage: apply_ddc_to_holosoma.sh <path-to-holosoma-checkout> [--force]}"
FORCE="${2:-}"   # --force = overlay even if the checkout is not at the pinned commit
DEST="$HOLO/src/holosoma/holosoma"

echo "== 1. verify DDC files here match ddc_src/SHA256SUMS =="
( cd "$SRC" && sha256sum -c SHA256SUMS ) || { echo "FAIL: local DDC files altered/corrupt"; exit 1; }

echo "== 2. verify the Holosoma checkout is at the pinned commit =="
if [ -d "$HOLO/.git" ]; then
  HEAD=$(git -C "$HOLO" rev-parse HEAD)
  if [ "$HEAD" != "$PIN" ]; then
    echo "MISMATCH: checkout is at $HEAD, not the pinned $PIN."
    echo "          Get it with:  git -C '$HOLO' fetch origin && git -C '$HOLO' checkout $PIN"
    if [ "$FORCE" != "--force" ]; then
      echo "          Refusing to overlay onto an unpinned checkout. Re-run with --force to override"
      echo "          (upstream drift may need minor adaptation — see TRAINING.md)."
      exit 1
    fi
    echo "          --force given: overlaying anyway."
  else
    echo "OK: at pinned commit $PIN"
  fi
else
  echo "NOTE: $HOLO is not a git checkout; cannot verify the commit (expected upstream = $PIN)."
  if [ "$FORCE" != "--force" ]; then
    echo "      Refusing to overlay onto an unverifiable checkout. Re-run with --force to override."
    exit 1
  fi
  echo "      --force given: overlaying anyway."
fi
[ -d "$DEST" ] || { echo "FAIL: $DEST not found — is '$HOLO' a Holosoma checkout?"; exit 1; }

echo "== 3. overlay the DDC files =="
while IFS= read -r rel; do
  mkdir -p "$DEST/$(dirname "$rel")"
  cp -v "$SRC/$rel" "$DEST/$rel"
done < <(cd "$SRC" && find . -type f ! -name SHA256SUMS | sed 's|^\./||')

echo "== done. Now train:  DATA=/path/to/data_stratified_900/train bash '$HERE/train_full.sh'  =="
