# Training Journey Notes

This note is not meant to replace the formal README. It records the practical training path behind the current Word2Vec project: what was tried first, what looked wrong, which questions came up during analysis, and why some runs were still useful even when the embedding space was not yet where it needed to be.

If more detail is needed for any run, the most reliable source is the corresponding `run_config.json` inside that checkpoint folder.

## Why This Note Exists

In this project, training loss, held-out SGNS loss, and qualitative semantic quality do not always move together.

Some runs were useful because they exposed a failure mode clearly:

- high-frequency words dominating the space
- semantic neighborhoods looking worse than the loss would suggest
- averaged embeddings hiding or distorting what was happening in `W_center` and `W_context`

The goal of this note is to document that process honestly.

## Example 1: Early No-Subsampling Baseline

Reference run:

- `checkpoints/20260314_200105_7hr`

This run is a good example of an early baseline that was important mainly because it made the problem visible.

Key characteristics from `run_config.json`:

- no explicit subsampling
- `max_vocab_size = 50000`
- `min_freq = 2`
- `num_negative_samples = 15`
- `learning_rate = 0.1`
- `window_size = 2`

What this run helped reveal:

- the embedding space was heavily influenced by very frequent words
- many words looked too close to each other
- qualitative plots suggested that the space was not forming clean semantic neighborhoods

This did not make the run useless. On the contrary, it established a clear baseline and showed why subsampling and tighter vocabulary control were needed.

Illustrative figure from that run:

![No-subsampling t-SNE example](readme_assets/no_subsampling_tsne_20260314_200105.png)

The figure is not presented as a benchmark result. It is included because it captures the practical impression from that stage: the space did not look cleanly organized enough to trust semantic nearest neighbors.

## Example 2: Validation-Aware Warm-Start Experiment

Reference run:

- `checkpoints/20260315_231058_bestbest`

This run represents a later stage where the pipeline was already more careful:

- smaller vocabulary
- subsampling enabled
- validation tracking enabled
- warm-start training from an earlier checkpoint

Key characteristics from `run_config.json`:

- `max_vocab_size = 10000`
- `min_freq = 5`
- `num_negative_samples = 5`
- `learning_rate = 0.02`
- `window_size = 2`
- `subsample_threshold = 1e-5`
- validation split and periodic validation loss
- warm start from `20260315_183537_con/final`

Why this run still mattered even though it was not the final answer:

- it showed a more disciplined training setup than the earliest runs
- it made loss tracking more informative
- it raised an important question: why can the numbers look better while the semantic neighborhoods still feel wrong?

That question turned out to be important. Later analysis showed that SGNS loss and qualitative semantic structure can diverge, and that looking only at the mean of `W_center` and `W_context` can be misleading.

In other words, this run was useful not because it solved the problem completely, but because it made the next problem easier to identify.

## What Changed Over Time

Across experiments, the project gradually moved from a simple training script toward a more inspectable workflow:

- added subsampling after seeing that frequent words were dominating the space
- saved run-specific vocabularies for reproducible evaluation
- added validation tracking for held-out SGNS loss
- added warm-start support for continuation experiments
- expanded qualitative inspection through notebook probes, nearest neighbors, PCA, t-SNE, and cosine heatmaps
- added stricter token filtering to reduce low-signal tokens in the training data

These additions did not make every later run automatically good, but they made the experiments easier to interpret.

## Current Position

The current project is in a more mature state than the earliest runs, but the central lesson remains:

- lower loss is useful
- lower loss is not sufficient
- qualitative embedding checks are still necessary

For that reason, the project now treats both kinds of evidence as important:

- optimization-oriented evidence such as training and validation SGNS loss
- representation-oriented evidence such as nearest neighbors, analogy probes, and geometry diagnostics

## Pending Analysis

The following runs were added later in the project after the training pipeline had been updated with:

- stricter token filtering
- an explicit learning-rate schedule
- validation-based best-checkpoint saving
- a stronger focus on checking `W_center`, not only averaged embeddings

### Run Placeholder: 20260316_015809

Status:

- later-stage experiment after the newer training changes

Small conclusion for now:

- this run belongs to the "new pipeline" stage rather than the early debugging stage
- it should be judged mainly by validation behavior plus qualitative `W_center` inspection
- a full conclusion should wait until its saved histories and nearest-neighbor probes are reviewed together

### Run Placeholder: 20260316_015223

Status:

- later-stage experiment with the newer schedule and preprocessing choices

Small conclusion for now:

- this run changed several things at once: `window_size`, number of negatives, learning-rate schedule, and token filtering
- direct comparison of raw SGNS loss against older runs is therefore misleading because the objective scale changed
- qualitative checks suggest that the run did not collapse in the same way as `20260315_231058_bestbest`, but high-frequency generic words were still too dominant in `W_center`
- the main lesson from this run is that better engineering around training does not automatically remove frequency bias from the learned space

