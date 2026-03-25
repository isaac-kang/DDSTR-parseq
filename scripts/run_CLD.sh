#!/bin/bash
# Mode CLD: Confidence-based Label Denoising
# Step 1: Extract error info  -- active env (ddstr)
# Step 2: LLM judge           -- vllm env (conda run)
# Step 3: Generate PL LMDB   -- active env (ddstr)
# Step 4: Train               -- active env (ddstr)
#
# Usage: conda activate ddstr && bash scripts/run_CLD.sh [--steps=CLD,4]
#   --steps=4       (default) training only
#   --steps=CLD     steps 1-3 (data prep)
#   --steps=CLD,4   everything
#   --steps=all     everything

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A STEP_ALIASES=( [CLD]="123" )
DEFAULT_STEPS="4"
source "$SCRIPT_DIR/_parse_steps.sh"

LLM_ENV="${LLM_ENV:-vllm}"

DATA_ROOT="$(eval echo "${DATA_ROOT:-~/data/STR/parseq/english/lmdb}")"
CHECKPOINT="${CHECKPOINT:-pretrained=parseq}"
LLM_MODEL_IDX="${LLM_MODEL_IDX:-4}"  # Qwen3-8B
DENOISE_OUTPUT="denoise_output"

mkdir -p "$DENOISE_OUTPUT"

if run_step 1; then
    echo "=== Step 1: Extract error info ==="
    _T0=$SECONDS
    python tools/denoise/extract_error_info.py "$CHECKPOINT" \
        --data_root "$DATA_ROOT" \
        --output "${DENOISE_OUTPUT}/error_info.tsv"
    elapsed 1
fi

if run_step 2; then
    echo ""
    echo "=== Step 2: LLM judge ==="
    _T0=$SECONDS
    conda run --no-capture-output -n "$LLM_ENV" \
        python tools/denoise/llm_judge.py \
        "${DENOISE_OUTPUT}/error_info.tsv" "$LLM_MODEL_IDX" \
        --output "${DENOISE_OUTPUT}/judge_results.tsv"
    elapsed 2
fi

if run_step 3; then
    echo ""
    echo "=== Step 3: Generate denoised LMDB ==="
    _T0=$SECONDS
    python tools/denoise/generate_pl.py \
        --error_info "${DENOISE_OUTPUT}/error_info.tsv" \
        --judge "${DENOISE_OUTPUT}/judge_results.tsv" \
        --data_root "$DATA_ROOT" \
        --output_root "${DATA_ROOT}/denoised"
    elapsed 3
fi

if run_step 4; then
    echo ""
    echo "=== Step 4: Train on denoised data ==="
    _T0=$SECONDS
    python train.py +experiment=parseq_CLD \
        mode=CLD \
        pl_root_dir="${DATA_ROOT}/denoised" \
        wandb.group=Quokka \
        wandb.name=CLD \
        trainer.devices=1 \
        "${PASS_ARGS[@]}"
    elapsed 4
fi
