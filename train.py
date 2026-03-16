import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Ensure project-root imports work even when executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.preprocess import (
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
from src.data_prep.dataset import build_batch_generator
from src.train.model import SkipGramModel
from src.train.trainer import PartialTrainingInterrupt, train_model


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_train_config.json"

CONFIG_SPECS = {
    "num_epochs": {
        "flags": ["--epochs", "--num-epochs"],
        "type": int,
        "help": "Number of training epochs",
    },
    "max_vocab_size": {
        "flags": ["--max-vocab-size"],
        "type": int,
        "help": "Maximum vocabulary size",
    },
    "embedding_dim": {
        "flags": ["--embedding-dim"],
        "type": int,
        "help": "Embedding dimension",
    },
    "min_freq": {
        "flags": ["--min-freq"],
        "type": int,
        "help": "Minimum token frequency to keep in the vocabulary",
    },
    "batch_size": {
        "flags": ["--batch-size"],
        "type": int,
        "help": "Training batch size",
    },
    "num_negative_samples": {
        "flags": ["--num-negative-samples"],
        "type": int,
        "help": "Number of negative samples per positive pair",
    },
    "norm_factor": {
        "flags": ["--norm-factor"],
        "type": float,
        "help": "Exponent used for the unigram negative-sampling distribution",
    },
    "seed": {
        "flags": ["--seed"],
        "type": int,
        "help": "Random seed",
    },
    "learning_rate": {
        "flags": ["--learning-rate"],
        "type": float,
        "help": "Peak learning rate",
    },
    "learning_rate_start": {
        "flags": ["--learning-rate-start"],
        "type": float,
        "help": "Initial learning rate before warmup",
    },
    "learning_rate_min": {
        "flags": ["--learning-rate-min"],
        "type": float,
        "help": "Minimum learning rate after decay",
    },
    "learning_rate_warmup_ratio": {
        "flags": ["--learning-rate-warmup-ratio"],
        "type": float,
        "help": "Fraction of training used for learning-rate warmup",
    },
    "window_size": {
        "flags": ["--window-size"],
        "type": int,
        "help": "Skip-gram context window size",
    },
    "subsample_threshold": {
        "flags": ["--subsample-threshold"],
        "type": float,
        "help": "Frequent-word subsampling threshold",
    },
    "remove_stopwords": {
        "flags": ["--remove-stopwords"],
        "type": "bool",
        "help": "Whether to remove stopwords during preprocessing",
    },
    "split": {
        "flags": ["--split"],
        "type": str,
        "help": "Dataset split used for training",
    },
    "validation_split": {
        "flags": ["--validation-split"],
        "type": str,
        "help": "Dataset split used for validation loss",
    },
    "validation_every": {
        "flags": ["--validation-every"],
        "type": int,
        "help": "Evaluate validation loss every N training steps",
    },
    "validation_max_sentences": {
        "flags": ["--validation-max-sentences"],
        "type": int,
        "help": "Maximum number of validation sentences to evaluate",
    },
    "checkpoint_every": {
        "flags": ["--checkpoint-every"],
        "type": int,
        "help": "Save the latest checkpoint every N training steps",
    },
}


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_config(config_path: Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_config_override_args(parser: argparse.ArgumentParser) -> None:
    for field, spec in CONFIG_SPECS.items():
        arg_type = parse_bool if spec["type"] == "bool" else spec["type"]
        parser.add_argument(
            *spec["flags"],
            dest=field,
            type=arg_type,
            default=None,
            help=f"{spec['help']} (overrides config)",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Skip-gram with Negative Sampling")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to training config JSON (default: {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)})",
    )
    add_config_override_args(parser)
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


def apply_cli_overrides(config: dict, args) -> dict:
    updated_config = dict(config)

    for field in CONFIG_SPECS:
        value = getattr(args, field, None)
        if value is not None:
            updated_config[field] = value

    return updated_config


def split_training_config(config: dict) -> tuple[dict, int]:
    training_config = dict(config)
    checkpoint_every = training_config.pop("checkpoint_every")
    return training_config, checkpoint_every


def main():
    args = parse_args()
    loaded_config = load_config(args.config)
    merged_config = apply_cli_overrides(loaded_config, args)
    hyperparams, checkpoint_every = split_training_config(merged_config)
    checkpoint_root = Path("checkpoints")

    np.random.seed(hyperparams["seed"])

    run_dir, latest_ckpt_dir, final_ckpt_dir, best_ckpt_dir = create_run_directories(checkpoint_root)
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
            best_ckpt_dir=best_ckpt_dir,
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
            if validation_loss_records:
                print(f"Best validation checkpoint available at: {best_ckpt_dir}")
            print(f"Partial loss history saved: {artifacts['loss_csv']}")
            print(f"Partial training plot saved: {artifacts['plot']}")
            print(f"Partial run summary saved: {artifacts['summary']}")
        else:
            print(f"Final checkpoint saved: {final_ckpt_dir}")
            if validation_loss_records:
                print(f"Best validation checkpoint saved: {best_ckpt_dir}")
            print(f"Loss history saved: {artifacts['loss_csv']}")
            print(f"Training plot saved: {artifacts['plot']}")
            print(f"Run summary saved: {artifacts['summary']}")


if __name__ == "__main__":
    main()
