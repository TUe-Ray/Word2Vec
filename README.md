# Word2Vec (NumPy) - JetBrains Task #1

This project implements **Skip-gram with Negative Sampling (SGNS)** in **pure NumPy**.
No deep learning frameworks are used in the training loop.

## What Is Implemented

- Data loading from `wikitext-2-raw-v1`
- Text normalization and tokenization
- Vocabulary building and ID encoding
- Skip-gram pair generation
- Negative sampling with unigram^0.75 distribution
- Full SGNS optimization loop:
   - forward pass
   - loss computation
   - backward gradients
   - parameter updates (`np.add.at` for repeated indices)
- Checkpoint saving (`W_center.npy`, `W_context.npy`) and training records

## Environment Setup

1. Create and activate an environment.

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

## Dataset Download

Download and cache WikiText-2 raw dataset:
```bash
python download_dataset.py
```

Expected output folder:
- `data/wikitext-2-raw-v1/`

## Training

Run training from the project root:
```bash
python train.py
```

Main hyperparameters are defined in `train.py` under `hyperparams`.

## Outputs

Each run creates a timestamped folder under `checkpoints/`:

- `checkpoints/<run_id>/run_config.json`
   - Stores training hyperparameters and checkpoint interval.
- `checkpoints/<run_id>/latest/W_center.npy`
- `checkpoints/<run_id>/latest/W_context.npy`
   - Periodically updated checkpoint.
- `checkpoints/<run_id>/final/W_center.npy`
- `checkpoints/<run_id>/final/W_context.npy`
   - Final trained embeddings.
- `checkpoints/<run_id>/loss_history.csv`
   - Per-step training loss.
- `checkpoints/<run_id>/training_loss.png`
   - Loss curve visualization.
- `checkpoints/<run_id>/run_summary.json`
   - Final loss, best loss, total steps.

## Load Embeddings

Example:
```python
from src.model import SkipGramModel

model = SkipGramModel(vocab_size=50000, embedding_dim=1024)
model.load_embeddings("checkpoints/<run_id>/final")

# Input (center) embeddings
W_in = model.W_center
# Output (context) embeddings
W_out = model.W_context
```

Note: `vocab_size` and `embedding_dim` passed to `SkipGramModel` must match the saved checkpoint.

## Known Limitations

- No subsampling of frequent words yet.
- No learning-rate scheduling (constant LR only).
- No intrinsic evaluation pipeline yet (e.g., word similarity/analogy); `src/eval.py` is currently empty.
- Training is implemented for clarity and correctness, not maximum speed.
- The current script reads all generated skip-gram pairs in memory.

## Quick Sanity Check

Run preprocessing check script:
```bash
python test_preprocessing.py
```