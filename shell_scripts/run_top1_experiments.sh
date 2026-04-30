#!/bin/bash

# Activate the virtual environment so python dependencies like open_clip are found
source .venv/bin/activate

# Configuration
TRIALS=50
RESULTS_DIR="results"
CAPTION_FILES=(
    "linus_branch/captions/mhist_with_captions.csv"
    "linus_branch/captions/mhist_with_captions_pruned.csv"
    "linus_branch/captions/mhist_with_captions_all_prompts.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=1_pruned.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=1_all_prompts.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=2_pruned.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=2_all_prompts.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=4_pruned.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=4_all_prompts.csv"
)

echo "----------------------------------------------------------------"
echo "Starting TOP-1 Comparative Multimodal Experiments"
echo "----------------------------------------------------------------"

for FILE in "${CAPTION_FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "================================================================"
        echo "Processing (TOP-1): $FILE"
        echo "================================================================"
        
        # 1. Finetuned Backbone + Top-1 Caption
        echo ">>> Running FINETUNED Backbone + Top-1 Caption..."
        python src/multimodal_experiment.py --caption_file "$FILE" --trials $TRIALS --use_top1 --results_dir "$RESULTS_DIR"
        
        # 2. Frozen Backbone + Top-1 Caption
        echo ">>> Running FROZEN Backbone + Top-1 Caption..."
        python src/multimodal_experiment.py --caption_file "$FILE" --trials $TRIALS --freeze_backbone --use_top1 --results_dir "$RESULTS_DIR"
        
    else
        echo "Warning: Caption file $FILE not found. Skipping."
    fi
done

echo "----------------------------------------------------------------"
echo "TOP-1 Experiments Completed!"
echo "----------------------------------------------------------------"
