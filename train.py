import argparse
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Ensure project-root imports work even when executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import normalize_text, tokenize, build_vocab, encode_sentences
from src.utils import load_wikitext_raw, save_run_config, save_training_records
from src.dataset import generate_pairs, SkipGramBatchGenerator
from src.model import SkipGramModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train Skip-gram with Negative Sampling")
    parser.add_argument(
        "--start-weight-run-id",
        type=str,
        default=None,
        help="Checkpoint run id to load as start weights (for example: 20260314_191627)",
    )
    parser.add_argument(
        "--start-weight-subdir",
        type=str,
        default="latest",
        choices=["latest", "final"],
        help="Checkpoint subdirectory to load from when --start-weight-run-id is provided",
    )
    return parser.parse_args()


args = parse_args()

# training parameters
hyperparams = {
    "num_epochs": 5,
    "max_vocab_size": 50000,
    "embedding_dim": 1024,
    "min_freq": 2,
    "batch_size": 256,
    "num_negative_samples": 15,
    "norm_factor": 0.75,
    "seed": 42,
    "learning_rate": 0.1,
    "window_size": 2,
    "split": "train",
}

# checkpoint settings
checkpoint_every = 500
checkpoint_root = Path("checkpoints")
start_weight_run_id = args.start_weight_run_id
start_weight_subdir = args.start_weight_subdir

# reproducibility
np.random.seed(hyperparams["seed"])

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

start_weight_dir = None
if start_weight_run_id:
    start_weight_dir = checkpoint_root / start_weight_run_id / start_weight_subdir

run_config = {
    **hyperparams,
    "checkpoint_every": checkpoint_every,
    "start_weight_run_id": start_weight_run_id,
    "start_weight_subdir": start_weight_subdir,
    "start_weight_dir": str(start_weight_dir) if start_weight_dir is not None else None,
}
save_run_config(run_dir, run_config)

print(f"Run folder: {run_dir}")

# preprocess the dataset and build the vocabulary
raw_lines = load_wikitext_raw(hyperparams["split"])
sentences = [tokenize(normalize_text(line)) for line in raw_lines]
vocab = build_vocab(
    sentences,
    min_freq=hyperparams["min_freq"],
    max_vocab=hyperparams["max_vocab_size"],
)
encoded_sentences = encode_sentences(sentences, vocab)



# generate training pairs and negative samples
pairs = generate_pairs(encoded_sentences, window_size=hyperparams["window_size"])
batch_gen = SkipGramBatchGenerator(
    pairs=pairs,
    vocab=vocab,
    batch_size=hyperparams["batch_size"],
    num_negatives=hyperparams["num_negative_samples"],
    norm_factor=hyperparams["norm_factor"],
    seed=hyperparams["seed"],
)

# initialize the Skip-Gram model
model = SkipGramModel(
    vocab_size=vocab['vocab_size'],
    embedding_dim=hyperparams["embedding_dim"]
)
if start_weight_dir is not None and start_weight_dir.exists():
    try:
        model.load_embeddings(start_weight_dir)
        print(f"Loaded start weights from: {start_weight_dir}")
    except (FileNotFoundError, ValueError) as err:
        print(f"Warning: could not load start weights from {start_weight_dir}: {err}")
        print("Proceeding with fresh random initialization.")
else:
    if start_weight_dir is None:
        print("No --start-weight-run-id provided. Proceeding with fresh random initialization.")
    else:
        print(f"Start weight directory not found: {start_weight_dir}")
        print("Proceeding with fresh random initialization.")
print(f"Model initialized with vocab size {model.vocab_size} and embedding dimension {model.embedding_dim}")
print("start model training")

global_step = 0
loss_records = []

for epoch in range(hyperparams["num_epochs"]):
    pbar = tqdm(batch_gen, desc=f"Epoch {epoch+1}", unit="batch", leave=True)
    
    for step, (center_id, context_id, negative_ids) in enumerate(pbar, start=1):
        global_step += 1

        loss, cache = model.forward(center_id, context_id, negative_ids)
        model.backward(cache, hyperparams["learning_rate"])
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
artifacts = save_training_records(run_dir, loss_records, global_step)

print(f"Final checkpoint saved: {final_ckpt_dir}")
print(f"Loss history saved: {artifacts['loss_csv']}")
print(f"Training plot saved: {artifacts['plot']}")
print(f"Run summary saved: {artifacts['summary']}")