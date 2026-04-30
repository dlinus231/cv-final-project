## Project Overview

This project investigates the diagnostic capabilities of CLIP-based transformer models (specifically QuiltNet-B-32) on the MHIST colorectal cancer dataset. 

## Quick Start: Interactive Demo

To see the final model's diagnostic performance and self-attention reasoning in action:

1.  **Clone & Setup:**
    ```bash
    git clone <repo-url>
    cd computer_vision_project
    ```

2.  **Install Dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r quilt_notebooks/requirements.txt
    ```

3.  **Run the Demo:**
    - Open **`quilt_notebooks/model_demo.ipynb`**.
    - Run the notebook cells to load the best multimodal weights discovered via Bayesian optimization.
    - The notebook will automatically sample random patient cases and display:
        - The pathology image.
        - **Vision Attention Maps**: Spatial regions the model prioritized.
        - **Text Attention Weights**: Semantic tokens in the clinical prompt that influenced the diagnosis.

## Advanced Usage

For technical details on training pipelines, caption generation, or baseline comparisons, please refer to the scripts in `quilt_src/` or the experimental logs in `quilt_results/`.