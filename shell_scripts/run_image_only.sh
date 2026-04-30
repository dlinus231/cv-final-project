#!/bin/bash

TRIALS=50
RESULTS_DIR="results"

echo "================================================================"
echo "Running IMAGE-ONLY Baseline experiments"
echo "================================================================"

# 1. Image-Only Finetuned
echo ">>> Running IMAGE-ONLY FINETUNED experiment..."
python src/image_only_experiment.py --trials $TRIALS --results_dir "$RESULTS_DIR"

# 2. Image-Only Frozen
echo ">>> Running IMAGE-ONLY FROZEN experiment..."
python src/image_only_experiment.py --trials $TRIALS --freeze_backbone --results_dir "$RESULTS_DIR"

echo "================================================================"
echo "Image-only experiments completed!"
echo "Check results in: results/image_only/"
echo "================================================================"
