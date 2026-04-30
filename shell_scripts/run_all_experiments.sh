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

echo "Starting batch multimodal experiments..."
echo "Total experiments: ${#CAPTION_FILES[@]}"

for FILE in "${CAPTION_FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "----------------------------------------------------------------"
        echo "Running: $FILE"
        echo "----------------------------------------------------------------"
        python src/multimodal_experiment.py --caption_file "$FILE" --trials $TRIALS
    else
        echo "Skipping missing file: $FILE"
    fi
done

echo "All experiments completed!"
