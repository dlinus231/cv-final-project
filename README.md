## Project Overview

This project investigates the diagnostic capabilities of CLIP-based transformer models on the MHIST breast cancer dataset. It compares two distinct training paradigms:
1.  **Multimodal Finetuning**: Adapting both the vision encoder and text encoder to the downstream task.
2.  **Image-Only Finetuning**: Freezing the pre-trained weights of the text encoder and training only the vision encoder.

The primary objective is to determine whether retaining the general semantic knowledge of the frozen text encoder provides a significant benefit (i.e., requiring less training data) for this specific pathology classification task.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd computer_vision_project
    ```

2.  **Install Dependencies:**
    It is recommended to use a Python virtual environment.
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r quilt_notebooks/requirements.txt
    ```

## Usage

### 1. Preprocessing
Generate the multimodal captions required for training:
```bash
python quilt_src/generate_captions.py --data_dir "quilt_data/mhist-crc-tme/mhist-crc-tme" --out_path "captions/mhist_with_captions_all_prompts.csv"
```

### 2. Training & Experimentation

To run the main experimental pipeline:
```bash
python quilt_src/multimodal_experiment.py
```

This will:
1.  Load the dataset annotations (`quilt_data/annotations.csv`).
2.  Split data into 70% training, 30% testing.
3.  Initialize the QuiltNet-B-32 model (frozen text, finetuned image).
4.  Train the model using Optuna for hyperparameter optimization.
5.  Evaluate the best model on the held-out test set.
6.  Generate a classification report and save the best model checkpoint.

To replicate the specific baseline experiments from the README:

**Baseline 1: Image-Only Finetuned**
```bash
python quilt_src/image_only_experiment.py --trials 50 --results_dir "quilt_results"
```

**Baseline 2: Image-Only Frozen**
```bash
python quilt_src/image_only_experiment.py --trials 50 --freeze_backbone --results_dir "quilt_results"
```

## Visualization

To analyze model predictions and attention maps on a single case:
```bash
python quilt_src/visualize_attention.py
```

## Interactive Demo

An interactive Jupyter Notebook is provided to demonstrate the final model's diagnostic reasoning through dual-encoder self-attention visualizations:

1.  Open `quilt_notebooks/model_demo.ipynb`.
2.  Run the cells to load the optimal weights and the dataset.
3.  Use the provided functions to sample random histology cases and see exactly where the model is looking in both the image and the clinical prompt.