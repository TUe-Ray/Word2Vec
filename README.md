# Word2Vec (NumPy) - JetBrains Test Task

This project implements **Skip-gram with Negative Sampling (SGNS)** for an internship programming task.

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
   - parameter updates
- Checkpoint saving (`W_center.npy`, `W_context.npy`) and training records

## Project Structure

```text
word2vec-in-numpy/
├── train.py                          # training entry point; loads JSON config and applies CLI overrides
├── download_dataset.py               # downloads and caches the WikiText-2 raw dataset
├── demo_word2vec_results.ipynb       # interactive notebook for nearest neighbors and qualitative probes
├── requirements.txt                  # Python dependencies
├── configs/
│   └── default_train_config.json     # default training hyperparameters
├── src/
│   ├── common/
│   │   └── utils.py                  # run config saving, checkpoint helpers, dataset loading utilities
│   ├── data_prep/
│   │   ├── preprocess.py             # normalization, tokenization, vocabulary building, subsampling
│   │   └── dataset.py                # skip-gram pair generation and batch sampling
│   ├── train/
│   │   ├── model.py                  # SGNS model, forward pass, loss, gradients, weight updates
│   │   └── trainer.py                # training loop, validation checks, checkpointing, LR schedule
│   └── eval/
│       ├── eval.py                   # held-out SGNS loss and nearest-neighbor evaluation
│       ├── visualize_embeddings.py   # PCA, t-SNE, and cosine heatmap visualization
│       └── demo_helpers.py           # notebook helpers for exploratory analysis
├── docs/
│   ├── sgns_derivation.md            # derivation note for the SGNS objective and gradients
│   ├── training_journey.md           # short write-up of the main experiment stages
│   └── readme_assets/                # figures used in the README
├── tests/                            # test folder for project checks
├── checkpoints/                      # saved runs, configs, loss histories, and embeddings
└── data/                             # cached dataset files
```

## Methodology

This project trains a NumPy implementation of **Skip-gram with Negative Sampling (SGNS)** on the local `wikitext-2-raw-v1` corpus.

For a step-by-step mathematical derivation of the SGNS objective and gradients, see [docs/sgns_derivation.md](docs/sgns_derivation.md).

The current training pipeline is:

1. normalize and tokenize raw WikiText-2 lines
2. build a capped vocabulary with `min_freq` / `max_vocab_size`
3. apply frequent-word subsampling
4. encode tokens with the exact run-specific vocabulary
5. generate skip-gram pairs with a configurable context window
6. train SGNS with unigram^0.75 negative sampling
7. monitor both training loss and periodic validation loss during training

## Results Analysis

### Early Baseline Without Subsampling

The earliest baseline made it clear that subsampling and vocabulary control were not optional. Without them, the embedding space became heavily dominated by high-frequency words, which made nearest-neighbor outputs difficult to trust.

`model_1`, an early no-subsampling run, illustrates this issue clearly. To keep the comparison fair, the `model_1` and `model_2` figures below use the same 40-word representative subset for t-SNE and the same six-word probe set for the cosine heatmap: `king`, `queen`, `man`, `woman`, `cat`, and `dog`.

<table>
  <colgroup>
    <col style="width: 60%;">
    <col style="width: 40%;">
  </colgroup>
  <tr>
    <td align="center"><strong>40-word t-SNE</strong></td>
    <td align="center"><strong>Small probe heatmap</strong></td>
  </tr>
  <tr>
    <td><img src="docs/readme_assets/model_1_tsne40/mean_latest_tsne.png" alt="model_1 40-word t-SNE" width="100%"></td>
    <td><img src="docs/readme_assets/model_1_probe/mean_latest_cosine_heatmap.png" alt="model_1 small probe heatmap" width="100%"></td>
  </tr>
</table>

Observation:

- the space is not cleanly separated into intuitive semantic groups
- in the heatmap, some local relationships are plausible, such as `man-woman` and `cat-dog` being relatively close
- however, the overall structure is still too mixed: `queen` is also very close to `dog`, and `king` remains too close to almost every other word in the probe
- this was an early sign that frequent-word dominance had to be addressed before nearest-neighbor analysis could be taken seriously

Taken together, these two figures suggest that the early pipeline was not yet producing a usable semantic space. The t-SNE projection is visually cluttered, and the geometry seems to be driven more by broad frequency effects than by clear semantic grouping. The small probe heatmap also shows a mixture of reasonable and unreasonable relationships: a few intuitive pairs appear, but unrelated words are still pulled together too strongly. At this stage, the main value of the run was diagnostic. It made the failure mode visible enough to justify later changes in preprocessing and training control.

### Controlled Training With Clearer Local Structure

`model_2` represents a later training setup in which subsampling, validation tracking, and a more controlled configuration had already been introduced. The figures below use the same visualization pipeline, the same 40-word representative subset for t-SNE, and the same six-word probe set for the heatmap as `model_1`, so that the comparison remains fair.

<table>
  <colgroup>
    <col style="width: 60%;">
    <col style="width: 40%;">
  </colgroup>
  <tr>
    <td align="center"><strong>40-word t-SNE</strong></td>
    <td align="center"><strong>Small probe heatmap</strong></td>
  </tr>
  <tr>
    <td><img src="docs/readme_assets/model_2_tsne40/mean_final_tsne.png" alt="model_2 40-word t-SNE" width="100%"></td>
    <td><img src="docs/readme_assets/model_2_probe/mean_final_cosine_heatmap.png" alt="model_2 small probe heatmap" width="100%"></td>
  </tr>
</table>

Observation:

- the geometry is more structured than in `model_1`, which suggests that the updated pipeline reduced the most obvious failure mode
- in the heatmap, `king-man`, `man-woman`, and `cat-dog` are all very close, showing stronger local structure than in `model_1`
- however, the probe still reveals a clear semantic problem: `queen` is grouped more closely with `cat` and `dog` than with `king` or `woman`
- this is why the project treats better loss and cleaner qualitative geometry as related but not interchangeable signals

Compared with `model_1`, `model_2` is clearly more interpretable. The t-SNE projection is less chaotic, and the six-word probe heatmap shows stronger local structure. At the same time, the comparison remains useful precisely because it is still imperfect: some relationships improved, but others became organized in the wrong way. This later setup therefore showed that lower SGNS loss was meaningful, but not sufficient as a stand-alone criterion for semantic quality, even after the training pipeline had improved.

Overall conclusion:

- the project moved from an obviously unstable early embedding space to a more interpretable later one
- the comparison between `model_1` and `model_2` shows that preprocessing and training controls made a real difference
- however, the later stage still does not justify treating loss as a direct measure of semantic quality, so qualitative diagnostics remain necessary

For a short write-up of the training journey, major experiment pivots, and why some runs were useful even when the embeddings were still imperfect, see [docs/training_journey.md](docs/training_journey.md).

## Get Started

For a quick interactive demo of the learned embeddings, open the notebook:

```bash
jupyter notebook demo_word2vec_results.ipynb
```

Use the notebook for exploratory analysis, custom word probes, analogy checks, and small diagnostic heatmaps.

### Quick Start

Clone the repository and navigate into it:

```bash
git clone https://github.com/TUe-Ray/Word2Vec
cd Word2Vec
```

### Installation

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

### Prepare Dataset

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

Default training hyperparameters are defined in [configs/default_train_config.json](configs/default_train_config.json).
`python train.py` loads that JSON config first, then applies any CLI overrides you pass.

The default training configuration now includes frequent-word subsampling via `subsample_threshold`.
Set it to `0` if you want to disable subsampling for an ablation run.

Example: override a few common hyperparameters directly from the command line:

```bash
python train.py --epochs 3 --embedding-dim 50 --window-size 2 --num-negative-samples 5
```

You can also point training to a different config file:

```bash
python train.py --config configs/default_train_config.json
```

All current training hyperparameters can be overridden from the CLI, including values such as:

- `--max-vocab-size`
- `--min-freq`
- `--batch-size`
- `--learning-rate`
- `--learning-rate-start`
- `--learning-rate-min`
- `--learning-rate-warmup-ratio`
- `--subsample-threshold`
- `--remove-stopwords true|false`
- `--validation-every`
- `--checkpoint-every`

The training loop also supports periodic validation tracking:

- `validation_split`: usually `validation`
- `validation_every`: evaluate validation loss every N training steps
- `validation_max_sentences`: cap validation sentences for faster monitoring

To load start weights from an existing checkpoint, pass a run id via CLI:

```bash
python train.py --start-weight-run-id 20260314_191627
```

Optional: choose checkpoint type (`latest` or `final`):

```bash
python train.py --start-weight-run-id 20260314_191627 --start-weight-subdir final
```

If `--start-weight-run-id` is not provided, training starts from fresh random initialization.
If the folder or files are missing, training also falls back to fresh random initialization.

## Outputs

Each run creates a timestamped folder at `checkpoints/<run_id>/` containing:

- `run_config.json`
   Stores the resolved training configuration for that run after loading the JSON config and applying any CLI overrides.

- `vocab.json`
   The exact vocabulary used during training for that run.

- `latest/`
   Contains `W_center.npy` and `W_context.npy` for the periodically updated checkpoint.

- `best/`
   Contains the checkpoint with the best validation loss seen so far during training.

- `final/`
   Contains `W_center.npy` and `W_context.npy` for the final trained embeddings.

- `loss_history.csv`
   Per-step training loss.

- `validation_loss_history.csv`
   Periodic validation loss measured during training.

- `training_loss.png`
   Training and validation loss visualization.

- `run_summary.json`
   Final/best training loss, final/best validation loss, total steps.

## Load Saved Embeddings (for Eval/Inference)

This section is for loading embeddings after training (for evaluation or analysis).
For warm-start training, use the CLI flags in the **Training** section.

Example:
```python
from src.train.model import SkipGramModel

model = SkipGramModel(vocab_size=50000, embedding_dim=1024)
model.load_embeddings("checkpoints/<run_id>/final")

# Input (center) embeddings
W_in = model.W_center
# Output (context) embeddings
W_out = model.W_context
```

Note: `vocab_size` and `embedding_dim` passed to `SkipGramModel` must match the saved checkpoint.

## Visualization

The core SGNS training loop is implemented in pure NumPy. scikit-learn is used only for post-training visualization utilities (PCA/t-SNE), not for model training.

Use the visualization script to inspect trained embeddings with PCA, t-SNE, and a cosine similarity heatmap:

```bash
python -m src.eval.visualize_embeddings --run-id 20260314_191627 --checkpoint-subdir latest
```

By default, the script:

- loads the requested checkpoint run
- rebuilds the matching vocabulary from `run_config.json` if `checkpoints/<run_id>/vocab.json` does not exist yet
- uses the mean of `W_center` and `W_context`
- prefers a curated set of representative nouns, verbs, and adjectives, then fills the remainder with frequent words
- writes outputs to `checkpoints/<run_id>/visualizations/`

Example with specific words:

```bash
python -m src.eval.visualize_embeddings --run-id 20260314_191627 --checkpoint-subdir latest --words king queen man woman city london paris
```

Useful flags:

- `--embedding-source center|context|mean`
- `--num-words 120`
- `--annotate-limit 40`
- `--heatmap-limit 30`

## Evaluation

Use `src/eval/eval.py` to measure held-out SGNS loss on validation/test splits and inspect nearest neighbors:

```bash
python -m src.eval.eval --run-id 20260314_191627 --checkpoint-subdir latest
```

This writes evaluation outputs to `checkpoints/<run_id>/evaluation/`:

- `heldout_loss_summary.json`
- `nearest_neighbors_mean.json` (or `center/context` depending on `--embedding-source`)

Example with custom query words:

```bash
python -m src.eval.eval --run-id 20260314_191627 --checkpoint-subdir latest --query-words king queen london paris --top-k-neighbors 10
```

For faster iteration during tuning, you can evaluate only the first part of each split:

```bash
python -m src.eval.eval --run-id 20260314_191627 --checkpoint-subdir latest --eval-splits validation --max-sentences 200
```

## Known Limitations

- Lower SGNS loss does not automatically imply better semantic neighborhoods. Held-out loss and qualitative embedding quality can diverge.
- The mean of `W_center` and `W_context` is not always the best semantic representation. In some runs it can hide, distort, or cancel useful structure.
- High-frequency words can still dominate the geometry of the space through hubness and broad contextual overlap, even when subsampling is enabled.
- Token filtering and vocabulary design still matter a lot. Small preprocessing changes can noticeably affect both validation loss and qualitative results.
- No word similarity or analogy benchmark dataset integration yet.
- Qualitative evaluation is still partly manual. The project does not yet include an automatic semantic early-stopping signal.
- Training is implemented for clarity and correctness, not maximum speed.
- The current script reads all generated skip-gram pairs in memory.

