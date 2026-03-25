#!/bin/bash
# Baseline: Standard PARSeq training
# Usage: bash scripts/run_baseline.sh

set -e

python train.py +experiment=parseq \
    mode=baseline \
    wandb.group=Quokka \
    wandb.name=baseline \
    trainer.devices=1 \
    "$@"
