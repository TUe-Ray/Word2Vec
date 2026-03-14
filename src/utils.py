import json
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from typing import List, Dict

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


def save_training_records(run_dir: Path, loss_records: List[Dict[str, Any]], global_step: int) -> Dict[str, Path]:
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
    plot_path = run_dir / "training_loss.png"
    summary_path = run_dir / "run_summary.json"

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
        plt.savefig(plot_path, dpi=200)
        plt.close()

        summary = {
            "total_steps": global_step,
            "num_records": len(loss_records),
            "final_loss": loss_records[-1]["loss"],
            "best_loss": min(row["loss"] for row in loss_records),
        }
    else:
        summary = {
            "total_steps": global_step,
            "num_records": 0,
            "final_loss": None,
            "best_loss": None,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loss history saved to {loss_csv_path}")
    print(f"Run summary saved to {summary_path}")
    if loss_records:
        print(f"Training plot saved to {plot_path}")

    return {
        "loss_csv": loss_csv_path,
        "plot": plot_path,
        "summary": summary_path,
    }
