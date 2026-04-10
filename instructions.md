# Python Development Guide: Standard Tooling

This project uses **uv** for high-performance dependency management and as a runner for our main toolchain (Ruff, Pyright, etc.). Follow these standards for consistency and efficiency.

---

## 🛠️ The Toolchain

1.  **Environment & Dependency Management**: [uv](https://github.com/astral-sh/uv)
2.  **Linting & Formatting**: [Ruff](https://github.com/astral-sh/ruff) (invoked via `uv ruff`)
3.  **Static Type Checking**: [Pyright](https://github.com/microsoft/pyright) (invoked via `uv run pyright`)
4.  **Hardware Optimization**: MPS (Metal Performance Shaders) for local macOS development.

---

## 🚀 Environment Management with `uv`

### Initial Setup
Initialize the project infrastructure and sync dependencies from `pyproject.toml`.

```bash
# Initialize a new project (if necessary)
uv init

# Sync virtual environment with lockfile
uv sync

# Add project-specific dependencies
uv add open_clip_torch torch torchvision
```

### Running Scripts
Always use `uv run` to execute scripts within the correct environment context.

```bash
uv run python main.py
```

---

## 🧹 Linting, Formatting & Type Checking

Ensure all code follows our quality standards using the following commands.

### Ruff: The All-in-One Python Tool
We use Ruff for both linting and formatting. It replaces `flake8`, `isort`, and `black`.

```bash
# Run the linter
uv ruff check .

# Apply safe auto-fixes
uv ruff check . --fix

# Format the entire codebase
uv ruff format .
```

### Pyright: Static Type Checking
For robust static analysis, always verify your types before commits.

```bash
# Performance-optimized type checking
uv run pyright .
```

> [!TIP]
> Use `uv run pyright --watch` during active development for instantaneous type feedback.

---

## 🔬 Working with Models (`open_clip`)

When integrating multi-modal models like **QuiltNet**, ensure you are using the correct preprocessing pipelines.

### Model Loading Reference
```python
import open_clip
import torch

# 1. Initialize model and transforms
# Reference: 'hf-hub:wisdomik/QuiltNet-B-32'
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:wisdomik/QuiltNet-B-32'
)
tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')

# 2. Configure Local Hardware Acceleration (MPS for Apple Silicon)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# 3. Model Inference Setup
model.eval()
with torch.no_grad():
    # Preprocess image and tokenize text
    # image_input = preprocess_val(image).unsqueeze(0).to(device)
    # text_input = tokenizer(["a description", "another description"]).to(device)
    
    # image_features = model.encode_image(image_input)
    # text_features = model.encode_text(text_input)
    pass
```

---

## ✅ Pre-commit Standard Checklist

Run these commands in order before submitting code for review:

1.  `uv ruff format .` (Consistency)
2.  `uv ruff check . --fix` (Logic & Style)
3.  `uv run pyright .` (Type Safety)
