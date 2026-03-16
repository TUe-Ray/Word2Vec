import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure project-root imports work even when executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    prepare_training_data,
    prepare_validation_data,
)
from src.common.utils import (
    build_run_config,
    create_run_directories,
    maybe_load_start_weights,
    resolve_start_weight_dir,
    save_run_config,
    save_vocab,
    save_training_records,
)
from src.data.dataset import build_batch_generator
from src.train.model import SkipGramModel
from src.train.trainer import PartialTrainingInterrupt, train_model



DEFAULT_HYPERPARAMS = {
    "num_epochs": 10,
    "max_vocab_size": 20000,
    "embedding_dim": 200,
    "min_freq": 5,
    "batch_size": 256,
    "num_negative_samples": 10,
    "norm_factor": 0.75,
    "seed": 42,
    "learning_rate": 0.05,
    "learning_rate_start": 0.025,
    "learning_rate_min": 0.005,
    "learning_rate_warmup_ratio": 0.1,
    "window_size": 4,
    "subsample_threshold": 5e-6,
    "split": "train",
    "validation_split": "validation",
    "validation_every": 500,
    "validation_max_sentences": 200,
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
    validation_pairs = prepare_validation_data(vocab, hyperparams)
    run_vocab_path = run_dir / "vocab.json"
    save_vocab(vocab, str(run_vocab_path))
    batch_gen = build_batch_generator(pairs, vocab, hyperparams)

    model = SkipGramModel(
        vocab_size=vocab["vocab_size"],
        embedding_dim=hyperparams["embedding_dim"],
    )
    maybe_load_start_weights(model, start_weight_dir)

    if args.start_weight_run_id:
        print(f"✓ Successfully continuing training from checkpoint: {args.start_weight_run_id}")
    print(f"Model initialized with vocab size {model.vocab_size} and embedding dimension {model.embedding_dim}")
    print("start model training")
    train_loss_records = []
    validation_loss_records = []
    global_step = 0
    interrupted = False

    try:
        train_loss_records, validation_loss_records, global_step = train_model(
            model=model,
            batch_gen=batch_gen,
            validation_pairs=validation_pairs,
            vocab=vocab,
            hyperparams=hyperparams,
            checkpoint_every=checkpoint_every,
            latest_ckpt_dir=latest_ckpt_dir,
        )
    except PartialTrainingInterrupt as exc:
        interrupted = True
        train_loss_records = exc.train_loss_records
        validation_loss_records = exc.validation_loss_records
        global_step = exc.global_step
        print("\nTraining interrupted by user. Saving partial progress...")
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving partial progress...")
    finally:
        # Always persist weights and records so progress is not lost on interruption.
        model.save_embeddings(latest_ckpt_dir)
        model.save_embeddings(final_ckpt_dir)
        artifacts = save_training_records(
            run_dir,
            train_loss_records,
            global_step,
            validation_loss_records=validation_loss_records,
        )

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
