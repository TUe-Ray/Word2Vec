import argparse
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Ensure project-root imports work even when executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    normalize_text,
    tokenize,
    build_vocab,
    subsample_frequent_words,
    encode_sentences,
)
from src.utils import (
    create_run_directories,
    load_wikitext_raw,
    resolve_start_weight_dir,
    save_run_config,
    save_vocab,
    save_training_records,
)
from src.dataset import generate_pairs, SkipGramBatchGenerator
from src.model import SkipGramModel



DEFAULT_HYPERPARAMS = {
    "num_epochs": 5,
    "max_vocab_size": 10000,
    "embedding_dim": 100,
    "min_freq": 5,
    "batch_size": 256,
    "num_negative_samples": 5,
    "norm_factor": 0.75,
    "seed": 42,
    "learning_rate": 0.05,
    "window_size": 5,
    "subsample_threshold": 1e-5,
    "split": "train",
}
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



def build_run_config(hyperparams, checkpoint_every, start_weight_run_id, start_weight_subdir, start_weight_dir):
    return {
        **hyperparams,
        "checkpoint_every": checkpoint_every,
        "start_weight_run_id": start_weight_run_id,
        "start_weight_subdir": start_weight_subdir,
        "start_weight_dir": str(start_weight_dir) if start_weight_dir is not None else None,
    }


def prepare_training_data(hyperparams):
    raw_lines = load_wikitext_raw(hyperparams["split"])
    sentences = [tokenize(normalize_text(line)) for line in raw_lines]
    vocab = build_vocab(
        sentences,
        min_freq=hyperparams["min_freq"],
        max_vocab=hyperparams["max_vocab_size"],
    )
    subsampled_sentences, subsample_stats = subsample_frequent_words(
        sentences,
        vocab,
        threshold=hyperparams.get("subsample_threshold", 0.0),
        seed=hyperparams["seed"],
    )
    print(
        "Subsampling stats: "
        f"tokens {subsample_stats['tokens_before']} -> {subsample_stats['tokens_after']} "
        f"(dropped {subsample_stats['dropped_tokens']}, "
        f"{subsample_stats['drop_ratio']:.2%}), "
        f"sentences {subsample_stats['sentences_before']} -> {subsample_stats['sentences_after']}"
    )
    encoded_sentences = encode_sentences(subsampled_sentences, vocab)
    pairs = generate_pairs(encoded_sentences, window_size=hyperparams["window_size"])
    return vocab, pairs


def build_batch_generator(pairs, vocab, hyperparams):
    return SkipGramBatchGenerator(
        pairs=pairs,
        vocab=vocab,
        batch_size=hyperparams["batch_size"],
        num_negatives=hyperparams["num_negative_samples"],
        norm_factor=hyperparams["norm_factor"],
        seed=hyperparams["seed"],
    )


def maybe_load_start_weights(model, start_weight_dir):
    if start_weight_dir is None:
        print("No --start-weight-run-id provided. Proceeding with fresh random initialization.")
        return

    if not start_weight_dir.exists():
        print(f"Start weight directory not found: {start_weight_dir}")
        print("Proceeding with fresh random initialization.")
        return

    try:
        model.load_embeddings(start_weight_dir)
        print(f"Loaded start weights from: {start_weight_dir}")
    except (FileNotFoundError, ValueError) as err:
        print(f"Warning: could not load start weights from {start_weight_dir}: {err}")
        print("Proceeding with fresh random initialization.")


def train_model(model, batch_gen, hyperparams, checkpoint_every, latest_ckpt_dir):
    global_step = 0
    loss_records = []

    for epoch in range(hyperparams["num_epochs"]):
        pbar = tqdm(batch_gen, desc=f"Epoch {epoch + 1}", unit="batch", leave=True)

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

    return loss_records, global_step


def main():
    args = parse_args()
    hyperparams = dict(DEFAULT_HYPERPARAMS)
    checkpoint_every = 500
    checkpoint_root = Path("checkpoints")

    np.random.seed(hyperparams["seed"])

    run_dir, latest_ckpt_dir, final_ckpt_dir = create_run_directories(checkpoint_root)
    start_weight_dir = resolve_start_weight_dir(
        checkpoint_root=checkpoint_root,
        run_id=args.start_weight_run_id,
        subdir=args.start_weight_subdir,
    )

    run_config = build_run_config(
        hyperparams=hyperparams,
        checkpoint_every=checkpoint_every,
        start_weight_run_id=args.start_weight_run_id,
        start_weight_subdir=args.start_weight_subdir,
        start_weight_dir=start_weight_dir,
    )
    save_run_config(run_dir, run_config)
    print(f"Run folder: {run_dir}")

    vocab, pairs = prepare_training_data(hyperparams)
    run_vocab_path = run_dir / "vocab.json"
    save_vocab(vocab, str(run_vocab_path))
    batch_gen = build_batch_generator(pairs, vocab, hyperparams)

    model = SkipGramModel(
        vocab_size=vocab["vocab_size"],
        embedding_dim=hyperparams["embedding_dim"],
    )
    maybe_load_start_weights(model, start_weight_dir)

    print(f"Model initialized with vocab size {model.vocab_size} and embedding dimension {model.embedding_dim}")
    print("start model training")
    loss_records = []
    global_step = 0
    interrupted = False

    try:
        loss_records, global_step = train_model(
            model=model,
            batch_gen=batch_gen,
            hyperparams=hyperparams,
            checkpoint_every=checkpoint_every,
            latest_ckpt_dir=latest_ckpt_dir,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving partial progress...")
    finally:
        # Always persist weights and records so progress is not lost on interruption.
        model.save_embeddings(latest_ckpt_dir)
        model.save_embeddings(final_ckpt_dir)
        artifacts = save_training_records(run_dir, loss_records, global_step)

        if interrupted:
            print(f"Partial checkpoint saved (latest): {latest_ckpt_dir}")
            print(f"Partial checkpoint saved (final): {final_ckpt_dir}")
            print(f"Partial loss history saved: {artifacts['loss_csv']}")
            print(f"Partial training plot saved: {artifacts['plot']}")
            print(f"Partial run summary saved: {artifacts['summary']}")
        else:
            print(f"Final checkpoint saved: {final_ckpt_dir}")
            print(f"Loss history saved: {artifacts['loss_csv']}")
            print(f"Training plot saved: {artifacts['plot']}")
            print(f"Run summary saved: {artifacts['summary']}")


if __name__ == "__main__":
    main()
