#!/usr/bin/env bash
# Reproduce all bootstrap evaluation results for the rebuttal.
#
# Usage:
#   bash rebuttal/evaluation_scripts/run_bootstrap.sh \
#       /path/to/jetnet_real.npy /path/to/jetnet_gen.npy \
#       /path/to/calo_real.npy   /path/to/calo_gen.npy
#
# Outputs JSON results to rebuttal/results/

set -euo pipefail

JETNET_REAL="${1:?Usage: $0 jetnet_real.npy jetnet_gen.npy calo_real.npy calo_gen.npy}"
JETNET_GEN="${2:?}"
CALO_REAL="${3:?}"
CALO_GEN="${4:?}"

OUTDIR="rebuttal/results"
mkdir -p "$OUTDIR"

echo "=== JetNet Bootstrap Evaluation ==="
uv run python -m BUFF.evaluation.run_jetnet_eval \
    --real-data "$JETNET_REAL" \
    --gen-data "$JETNET_GEN" \
    --n-bootstrap 100 \
    --output "$OUTDIR/jetnet_bootstrap.json"

echo ""
echo "=== CaloChallenge Bootstrap Evaluation ==="
uv run python -m BUFF.evaluation.run_calo_eval \
    --real-data "$CALO_REAL" \
    --gen-data "$CALO_GEN" \
    --n-bootstrap 100 \
    --output "$OUTDIR/calo_bootstrap.json"

echo ""
echo "=== CaloChallenge Discriminator Test ==="
uv run python -m BUFF.evaluation.run_calo_eval \
    --real-data "$CALO_REAL" \
    --gen-data "$CALO_GEN" \
    --run-discriminator \
    --n-bootstrap 100 \
    --output "$OUTDIR/calo_discriminator.json"

echo ""
echo "Done. Results in $OUTDIR/"
