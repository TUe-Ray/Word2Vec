import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.preprocess import normalize_text, tokenize, build_vocab, encode_sentences
from src.utils import load_wikitext_raw
from src.dataset import generate_pairs, SkipGramBatchGenerator
from src.model import SkipGramModel

# training parameters
num_epochs = 5
max_vocab_size = 50000
embedding_dim = 100
min_freq = 2
batch_size = 256
num_negative_samples = 15
norm_factor = 0.75
seed = 42
learning_rate = 0.1
window_size = 2
split = "train"

# checkpoint settings
checkpoint_every = 500
checkpoint_root = Path("checkpoints")

# reproducibility
np.random.seed(seed)

# create unique run directory based on start timestamp
run_id_base = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = checkpoint_root / run_id_base
suffix = 1
while run_dir.exists():
    run_dir = checkpoint_root / f"{run_id_base}_{suffix:02d}"
    suffix += 1

latest_ckpt_dir = run_dir / "latest"
final_ckpt_dir = run_dir / "final"
run_dir.mkdir(parents=True, exist_ok=False)

run_config = {
    "num_epochs": num_epochs,
    "max_vocab_size": max_vocab_size,
    "embedding_dim": embedding_dim,
    "min_freq": min_freq,
    "batch_size": batch_size,
    "num_negative_samples": num_negative_samples,
    "norm_factor": norm_factor,
    "seed": seed,
    "learning_rate": learning_rate,
    "window_size": window_size,
    "split": split,
    "checkpoint_every": checkpoint_every,
}

with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2, ensure_ascii=False)

print(f"Run folder: {run_dir}")

# preprocess the dataset and build the vocabulary
raw_lines = load_wikitext_raw(split)
sentences = [tokenize(normalize_text(line)) for line in raw_lines]
vocab = build_vocab(sentences, min_freq=min_freq, max_vocab=max_vocab_size)
encoded_sentences = encode_sentences(sentences, vocab)



# generate training pairs and negative samples
pairs = generate_pairs(encoded_sentences, window_size=window_size)
batch_gen = SkipGramBatchGenerator(
    pairs=pairs,
    vocab=vocab,
    batch_size=batch_size,
    num_negatives=num_negative_samples,
    norm_factor=norm_factor,
    seed=seed,
)

# initialize the Skip-Gram model
model = SkipGramModel(vocab_size=vocab['vocab_size'], embedding_dim=embedding_dim, activation_type='sigmoid')
print(f"Model initialized with vocab size {model.vocab_size} and embedding dimension {model.embedding_dim}")
print("start model training")

global_step = 0
loss_records = []

for epoch in range(num_epochs):
    pbar = tqdm(batch_gen, desc=f"Epoch {epoch+1}", unit="batch", leave=True)
    
    for step, (center_id, context_id, negative_ids) in enumerate(pbar, start=1):
        global_step += 1

        loss, cache = model.forward(center_id, context_id, negative_ids)
        model.backward(cache, learning_rate)
        model.update()

        loss_val = float(loss)
        loss_records.append(
            {
                "global_step": global_step,
                "epoch": epoch + 1,
                "step_in_epoch": step,
                "loss": loss_val,
            }
        )


        if step % 10 == 0:
            pbar.set_postfix(loss=f"{loss_val:.6f}")

        if global_step % checkpoint_every == 0:
            model.save_embeddings(latest_ckpt_dir)
            pbar.write(f"Checkpoint updated at step {global_step}: {latest_ckpt_dir}")

# final save
model.save_embeddings(final_ckpt_dir)

loss_csv_path = run_dir / "loss_history.csv"
with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["global_step", "epoch", "step_in_epoch", "loss"])
    writer.writeheader()
    writer.writerows(loss_records)

if loss_records:
    steps = [row["global_step"] for row in loss_records]
    losses = [row["loss"] for row in loss_records]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Batch loss", alpha=0.5, linewidth=1)

    smooth_window = min(200, len(losses))
    if smooth_window >= 5:
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        smooth_losses = np.convolve(losses, kernel, mode="valid")
        smooth_steps = steps[smooth_window - 1 :]
        plt.plot(smooth_steps, smooth_losses, label=f"Moving average ({smooth_window})", linewidth=2)

    plt.title("Training Loss Curve")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = run_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    summary = {
        "total_steps": global_step,
        "num_records": len(loss_records),
        "final_loss": loss_records[-1]["loss"],
        "best_loss": min(row["loss"] for row in loss_records),
    }
else:
    plot_path = run_dir / "training_loss.png"
    summary = {
        "total_steps": global_step,
        "num_records": 0,
        "final_loss": None,
        "best_loss": None,
    }

with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Final checkpoint saved: {final_ckpt_dir}")
print(f"Loss history saved: {loss_csv_path}")
print(f"Training plot saved: {plot_path}")
print(f"Run summary saved: {run_dir / 'run_summary.json'}")