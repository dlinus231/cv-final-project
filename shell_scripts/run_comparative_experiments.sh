#!/bin/bash

# Array of caption files to test
CAPTION_FILES=(
    "linus_branch/mhist_with_captions.csv"
    "linus_branch/mhist_with_captions_pruned.csv"
    "linus_branch/captions/mhist_with_captions_all_prompts.csv"
    "linus_branch/captions/mhist_with_captions_pruned.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=1_all_prompts.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=1_pruned.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=2_all_prompts.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=2_pruned.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=4_all_prompts.csv"
    "linus_branch/captions/mhist_with_captions_num_unfrozen=4_pruned.csv"
)

# Number of trials per experiment
TRIALS=50

# Results base directory
RESULTS_DIR="results"

echo "----------------------------------------------------------------"
echo "Starting Comparative Multimodal Experiments"
echo "----------------------------------------------------------------"

# 0. Run Image-Only Baseline Experiments
echo ">>> Running IMAGE-ONLY FINETUNED Baseline Experiment..."
python src/image_only_experiment.py --trials $TRIALS --results_dir "$RESULTS_DIR"

echo ">>> Running IMAGE-ONLY FROZEN Baseline Experiment..."
python src/image_only_experiment.py --trials $TRIALS --freeze_backbone --results_dir "$RESULTS_DIR"

for FILE in "${CAPTION_FILES[@]}"; do


    if [ -f "$FILE" ]; then
        echo "================================================================"
        echo "Processing: $FILE"
        echo "================================================================"
        
        # 1. Run Finetuned Backbone Experiment
        echo ">>> Running FINETUNED Backbone Experiment..."
        python src/multimodal_experiment.py --caption_file "$FILE" --trials $TRIALS --results_dir "$RESULTS_DIR"
        
        # 2. Run Frozen Backbone Experiment
        echo ">>> Running FROZEN Backbone Experiment..."
        python src/multimodal_experiment.py --caption_file "$FILE" --trials $TRIALS --freeze_backbone --results_dir "$RESULTS_DIR"
        
    else
        echo "Skipping missing file: $FILE"
    fi
done

echo "----------------------------------------------------------------"
echo "All comparative experiments completed!"
echo "Results are organized in: $RESULTS_DIR/finetuned/ and $RESULTS_DIR/frozen/"
echo "----------------------------------------------------------------"
