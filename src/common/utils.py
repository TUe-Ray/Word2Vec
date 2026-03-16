import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
from typing import List, Dict


def create_run_directories(checkpoint_root: Path) -> tuple[Path, Path, Path]:
    """Create a unique run directory and return (run_dir, latest_dir, final_dir)."""
    checkpoint_root = Path(checkpoint_root)
    run_id_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = checkpoint_root / run_id_base
    suffix = 1

    while run_dir.exists():
        run_dir = checkpoint_root / f"{run_id_base}_{suffix:02d}"
        suffix += 1

    latest_ckpt_dir = run_dir / "latest"
    final_ckpt_dir = run_dir / "final"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, latest_ckpt_dir, final_ckpt_dir


def resolve_start_weight_dir(checkpoint_root: Path, run_id: str | None, subdir: str) -> Path | None:
    """Return checkpoint directory for warm start, or None when no run id is provided."""
    if not run_id:
        return None
    return Path(checkpoint_root) / run_id / subdir


def build_run_config(
    hyperparams: Dict[str, Any],
    checkpoint_every: int,
    start_weight_run_id: str | None,
    start_weight_subdir: str,
    start_weight_dir: Path | None,
) -> Dict[str, Any]:
    return {
        **hyperparams,
        "checkpoint_every": checkpoint_every,
        "start_weight_run_id": start_weight_run_id,
        "start_weight_subdir": start_weight_subdir,
        "start_weight_dir": str(start_weight_dir) if start_weight_dir is not None else None,
    }


def maybe_load_start_weights(model, start_weight_dir: Path | None) -> None:
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

# General utility functions for dataset loading, vocabulary saving/loading, etc.

def load_wikitext_raw(split: str = 'train') -> List[str]:
    """
    Load wikitext-2-raw-v1 dataset from disk.
    
    Args:
        split: Dataset split - 'train', 'validation', or 'test'
    
    Returns:
        List of text lines from the specified split
    """
    dataset_path = Path('data/wikitext-2-raw-v1')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(dataset_path))
        if split not in ds:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")
        
        # Extract text from the dataset
        texts = ds[split]['text']
        # Filter out empty lines
        return [text for text in texts if text.strip()]
    except Exception as e:
        raise RuntimeError(f"Error loading wikitext dataset: {e}")


def save_vocab(vocab: Dict, filepath: str = 'data/vocab.json') -> None:
    """
    Save vocabulary to a JSON file.
    
    Args:
        vocab: Dictionary with keys 'word2id', 'id2word', 'word_freq', 'vocab_size'
        filepath: Path to save the vocabulary file
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert id2word keys to strings for JSON serialization
    vocab_to_save = {
        'word2id': vocab['word2id'],
        'id2word': {str(k): v for k, v in vocab['id2word'].items()},
        'word_freq': vocab['word_freq'],
        'vocab_size': vocab['vocab_size']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"Vocabulary saved to {output_path}")


def load_vocab(filepath: str = 'data/vocab.json') -> Dict:
    """
    Load vocabulary from a JSON file.
    
    Args:
        filepath: Path to the vocabulary file
    
    Returns:
        Dictionary with keys 'word2id', 'id2word', 'word_freq', 'vocab_size'
    """
    vocab_path = Path(filepath)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # Convert id2word keys back to integers
    vocab_data['id2word'] = {int(k): v for k, v in vocab_data['id2word'].items()}
    
    return vocab_data


def save_run_config(run_dir: Path, run_config: Dict[str, Any]) -> Path:
    """
    Save training run configuration to JSON.

    Args:
        run_dir: Directory for the current training run
        run_config: Hyperparameters and run metadata

    Returns:
        Path to the saved config file
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print(f"Run config saved to {config_path}")
    return config_path


def save_training_records(
    run_dir: Path,
    loss_records: List[Dict[str, Any]],
    global_step: int,
    validation_loss_records: List[Dict[str, Any]] | None = None,
) -> Dict[str, Path]:
    """
    Save loss history CSV, training loss plot, and run summary JSON.

    Args:
        run_dir: Directory for the current training run
        loss_records: List of per-step loss entries
        global_step: Last global training step

    Returns:
        Dictionary containing output artifact paths
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    loss_csv_path = run_dir / "loss_history.csv"
    validation_csv_path = run_dir / "validation_loss_history.csv"
    plot_path = run_dir / "training_loss.png"
    summary_path = run_dir / "run_summary.json"
    validation_loss_records = validation_loss_records or []

    with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "epoch", "step_in_epoch", "loss"])
        writer.writeheader()
        writer.writerows(loss_records)

    with open(validation_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "epoch", "step_in_epoch", "loss"])
        writer.writeheader()
        writer.writerows(validation_loss_records)

    if loss_records:
        import matplotlib.pyplot as plt

        steps = [row["global_step"] for row in loss_records]
        losses = [row["loss"] for row in loss_records]

        fig, ax_train = plt.subplots(figsize=(10, 5))
        legend_lines = []
        legend_labels = []

        line = ax_train.plot(
            steps,
            losses,
            label="Batch loss",
            alpha=0.5,
            linewidth=1,
            color="tab:blue",
        )[0]
        legend_lines.append(line)
        legend_labels.append(line.get_label())

        smooth_window = min(200, len(losses))
        if smooth_window >= 5:
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            smooth_losses = np.convolve(losses, kernel, mode="valid")
            smooth_steps = steps[smooth_window - 1 :]
            line = ax_train.plot(
                smooth_steps,
                smooth_losses,
                label=f"Training moving average ({smooth_window})",
                linewidth=2,
                color="tab:orange",
            )[0]
            legend_lines.append(line)
            legend_labels.append(line.get_label())

        if validation_loss_records:
            validation_steps = [row["global_step"] for row in validation_loss_records]
            validation_losses = [row["loss"] for row in validation_loss_records]
            ax_val = ax_train.twinx()
            line = ax_val.plot(
                validation_steps,
                validation_losses,
                label="Validation loss",
                linewidth=2,
                marker="o",
                markersize=3,
                color="tab:red",
            )[0]
            ax_val.set_ylabel("Validation loss", color="tab:red")
            ax_val.tick_params(axis="y", labelcolor="tab:red")
            legend_lines.append(line)
            legend_labels.append(line.get_label())

        ax_train.set_title("Training and Validation Loss")
        ax_train.set_xlabel("Global step")
        ax_train.set_ylabel("Training loss", color="tab:blue")
        ax_train.tick_params(axis="y", labelcolor="tab:blue")
        ax_train.grid(alpha=0.3)
        ax_train.legend(legend_lines, legend_labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        summary = {
            "total_steps": global_step,
            "num_train_records": len(loss_records),
            "num_validation_records": len(validation_loss_records),
            "final_train_loss": loss_records[-1]["loss"],
            "best_train_loss": min(row["loss"] for row in loss_records),
            "final_validation_loss": validation_loss_records[-1]["loss"] if validation_loss_records else None,
            "best_validation_loss": min(row["loss"] for row in validation_loss_records) if validation_loss_records else None,
        }
    else:
        summary = {
            "total_steps": global_step,
            "num_train_records": 0,
            "num_validation_records": len(validation_loss_records),
            "final_train_loss": None,
            "best_train_loss": None,
            "final_validation_loss": validation_loss_records[-1]["loss"] if validation_loss_records else None,
            "best_validation_loss": min(row["loss"] for row in validation_loss_records) if validation_loss_records else None,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loss history saved to {loss_csv_path}")
    print(f"Validation loss history saved to {validation_csv_path}")
    print(f"Run summary saved to {summary_path}")
    if loss_records:
        print(f"Training plot saved to {plot_path}")

    return {
        "loss_csv": loss_csv_path,
        "validation_csv": validation_csv_path,
        "plot": plot_path,
        "summary": summary_path,
    }
