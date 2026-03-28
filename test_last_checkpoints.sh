#!/bin/bash
# Find all last.ckpt under a given experiment directory and run test_pl.py on each.
# Usage: ./test_pl_last_checkpoints.sh <experiment_dir> [extra_args...]
# Example: ./test_pl_last_checkpoints.sh outputs/Octopus/train_pl-on-real_pl --cased

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_dir> [extra_args...]"
    echo "  Finds all last.ckpt files under <experiment_dir> and runs test_pl.py on each."
    exit 1
fi

EXP_DIR="$1"
shift
EXTRA_ARGS="$@"

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Directory '$EXP_DIR' not found"
    exit 1
fi

CKPTS=$(find "$EXP_DIR" -name "last.ckpt" -type f | sort)

if [ -z "$CKPTS" ]; then
    echo "No last.ckpt files found under $EXP_DIR"
    exit 1
fi

NUM_CKPTS=$(echo "$CKPTS" | wc -l)
echo "Found $NUM_CKPTS last.ckpt(s) under $EXP_DIR:"
echo "$CKPTS"
echo ""

IDX=0
for ckpt in $CKPTS; do
    IDX=$((IDX + 1))
    # Extract run directory name (e.g. 2026-03-24_00-40-46)
    RUN_DIR=$(dirname "$(dirname "$ckpt")")
    RUN_NAME=$(basename "$RUN_DIR")
    echo "========================================"
    echo "[$IDX/$NUM_CKPTS] $RUN_NAME"
    echo "  $ckpt"
    echo "========================================"
    python test.py "$ckpt" $EXTRA_ARGS
    echo ""
done

# ── Summary ──
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"

EXPERIMENT=$(echo "$EXP_DIR" | sed 's|.*outputs/||' | sed 's|/*$||')
echo "Experiment: $EXPERIMENT"
echo ""

printf "%-25s  %-70s  %10s\n" "Run" "Checkpoint" "bench_sub"
printf "%-25s  %-70s  %10s\n" "-------------------------" "----------------------------------------------------------------------" "----------"

for ckpt in $CKPTS; do
    LOG="${ckpt}.log.txt"
    RUN_DIR=$(dirname "$(dirname "$ckpt")")
    RUN_NAME=$(basename "$RUN_DIR")

    if [ -f "$LOG" ]; then
        bench_sub=$(awk '/^Benchmark \(Subset\) set:/{found=1} found && /\| Combined/{print $6; exit}' "$LOG")
        bench_sub=${bench_sub:-"-"}
    else
        bench_sub="(no log)"
    fi

    printf "%-25s  %-70s  %10s\n" "$RUN_NAME" "$ckpt" "$bench_sub"
done
