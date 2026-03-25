#!/bin/bash
# Mode CCD: Confusion-aware Class Decomposition
# Step 1: Build confusion matrix + create decomposed LMDB
# Step 2: Train with extended charset
#
# Usage: conda activate ddstr && bash scripts/run_CCD.sh [--steps=CCD,2]
#   --steps=2       (default) training only
#   --steps=CCD     step 1 (data prep)
#   --steps=CCD,2   everything
#   --steps=all     everything

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A STEP_ALIASES=( [CCD]="1" )
DEFAULT_STEPS="2"
source "$SCRIPT_DIR/_parse_steps.sh"

DATA_ROOT="$(eval echo "${DATA_ROOT:-~/data/STR/parseq/english/lmdb}")"
CONFUSION_OUTPUT="confusion_pl_output"
CHECKPOINT="${CHECKPOINT:-pretrained=parseq}"

if run_step 1; then
    echo "=== Step 1: Build confusion matrix & generate decomposed LMDB ==="
    _T0=$SECONDS
    python tools/confusion_and_pl.py \
        --checkpoint "$CHECKPOINT" \
        --data_root "$DATA_ROOT" \
        --output_dir "$CONFUSION_OUTPUT" \
        --min_rate 0.001 \
        --pl_datasets train/real \
        --pl_output_root "${DATA_ROOT}/decomposed"
    elapsed 1
fi

if run_step 2; then
    echo ""
    echo "=== Step 2: Train with extended charset ==="
    _T0=$SECONDS
    python train.py +experiment=parseq_CCD \
        mode=CCD \
        unicode_mapping="${CONFUSION_OUTPUT}/unicode_mapping.json" \
        pl_root_dir="${DATA_ROOT}/decomposed" \
        wandb.group=Quokka \
        wandb.name=CCD \
        trainer.devices=1 \
        "${PASS_ARGS[@]}"
    elapsed 2
fi
