import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ensure project-root imports work even when executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.preprocess import build_vocab, normalize_text, tokenize
from src.common.utils import load_vocab, load_wikitext_raw, save_vocab


SPECIAL_TOKENS = {"<PAD>", "<UNK>"}
DEFAULT_REPRESENTATIVE_WORDS = [
    "king",
    "queen",
    "man",
    "woman",
    "child",
    "family",
    "city",
    "country",
    "world",
    "river",
    "island",
    "school",
    "church",
    "team",
    "company",
    "government",
    "market",
    "system",
    "music",
    "science",
    "history",
    "war",
    "game",
    "film",
    "work",
    "play",
    "write",
    "build",
    "create",
    "support",
    "produce",
    "become",
    "include",
    "move",
    "win",
    "lose",
    "large",
    "small",
    "major",
    "public",
    "international",
    "political",
    "economic",
    "national",
    "early",
    "modern",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained word embeddings from a checkpoint run.")
    parser.add_argument("--run-id", type=str, required=True, help="Checkpoint run id, for example: 20260314_191627")
    parser.add_argument(
        "--checkpoint-subdir",
        type=str,
        default="final",
        choices=["latest", "final"],
        help="Checkpoint subdirectory to load from",
    )
    parser.add_argument(
        "--embedding-source",
        type=str,
        default="mean",
        choices=["center", "context", "mean"],
        help="Which embeddings to visualize",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=None,
        help="Specific vocabulary words to visualize. If omitted, the script uses the most frequent words.",
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=120,
        help="How many high-frequency words to visualize when --words is not provided",
    )
    parser.add_argument(
        "--annotate-limit",
        type=int,
        default=40,
        help="Maximum number of labels to draw on PCA and t-SNE plots",
    )
    parser.add_argument(
        "--heatmap-limit",
        type=int,
        default=30,
        help="Maximum number of words shown in the cosine similarity heatmap",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for t-SNE reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional external directory for visualization outputs and rebuilt vocab cache",
    )
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_dir(run_dir: Path, requested_subdir: str) -> Path:
    requested_dir = run_dir / requested_subdir
    if requested_dir.exists():
        return requested_dir

    fallback = "latest" if requested_subdir == "final" else "final"
    fallback_dir = run_dir / fallback
    if fallback_dir.exists():
        print(f"Requested checkpoint '{requested_subdir}' not found. Falling back to '{fallback}'.")
        return fallback_dir

    raise FileNotFoundError(
        f"Checkpoint directory not found. Tried: {requested_dir} and {fallback_dir}"
    )


def load_or_build_vocab(run_dir: Path, run_config: dict, cache_dir: Path | None = None) -> dict:
    run_vocab_path = run_dir / "vocab.json"
    if run_vocab_path.exists():
        print(f"Loading saved vocab from: {run_vocab_path}")
        return load_vocab(str(run_vocab_path))

    print("Saved run-specific vocab not found. Rebuilding vocab from the dataset split in run_config.json.")
    raw_lines = load_wikitext_raw(run_config["split"])
    sentences = [tokenize(normalize_text(line)) for line in raw_lines]
    vocab = build_vocab(
        sentences,
        min_freq=run_config["min_freq"],
        max_vocab=run_config["max_vocab_size"],
    )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_vocab_path = cache_dir / "vocab.json"
        save_vocab(vocab, str(cached_vocab_path))
    else:
        save_vocab(vocab, str(run_vocab_path))
    return vocab


def load_embedding_matrix(checkpoint_dir: Path, embedding_source: str) -> np.ndarray:
    w_center_path = checkpoint_dir / "W_center.npy"
    w_context_path = checkpoint_dir / "W_context.npy"

    if not w_center_path.exists():
        raise FileNotFoundError(f"Center embeddings not found: {w_center_path}")
    if not w_context_path.exists():
        raise FileNotFoundError(f"Context embeddings not found: {w_context_path}")

    w_center = np.load(w_center_path)
    w_context = np.load(w_context_path)

    if w_center.shape != w_context.shape:
        raise ValueError(
            f"Embedding shapes do not match: center={w_center.shape}, context={w_context.shape}"
        )

    if embedding_source == "center":
        return w_center
    if embedding_source == "context":
        return w_context
    return (w_center + w_context) / 2.0


def normalize_requested_word(raw_word: str) -> str | None:
    tokens = tokenize(normalize_text(raw_word))
    if len(tokens) != 1:
        return None
    return tokens[0]


def select_words(vocab: dict, requested_words: list[str] | None, num_words: int) -> tuple[list[str], list[str]]:
    word2id = vocab["word2id"]
    word_freq = vocab["word_freq"]

    if requested_words:
        selected = []
        missing = []
        seen = set()

        for raw_word in requested_words:
            word = normalize_requested_word(raw_word)
            if word is None:
                missing.append(raw_word)
                continue
            if word in seen:
                continue
            seen.add(word)
            if word in word2id and word not in SPECIAL_TOKENS:
                selected.append(word)
            else:
                missing.append(raw_word)

        return selected, missing

    selected = [word for word in DEFAULT_REPRESENTATIVE_WORDS if word in word2id]
    if len(selected) < num_words:
        ranked_words = sorted(
            (word for word in word2id if word not in SPECIAL_TOKENS),
            key=lambda word: (-word_freq.get(word, 0), word),
        )
        for word in ranked_words:
            if word in selected:
                continue
            selected.append(word)
            if len(selected) >= num_words:
                break

    return selected[:num_words], []


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def build_word_matrix(embedding_matrix: np.ndarray, vocab: dict, words: list[str]) -> np.ndarray:
    word_ids = [vocab["word2id"][word] for word in words]
    return embedding_matrix[word_ids]


def plot_projection(
    coords: np.ndarray,
    words: list[str],
    title: str,
    output_path: Path,
    annotate_limit: int,
):
    plt.figure(figsize=(12, 9))
    plt.scatter(coords[:, 0], coords[:, 1], s=28, alpha=0.75)

    for idx, word in enumerate(words[:annotate_limit]):
        plt.annotate(word, (coords[idx, 0], coords[idx, 1]), fontsize=8, alpha=0.85)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def create_pca_plot(word_matrix: np.ndarray, words: list[str], output_path: Path, annotate_limit: int):
    if len(words) < 2:
        print("Skipping PCA: need at least 2 words.")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(word_matrix)
    explained = pca.explained_variance_ratio_
    title = f"PCA of embeddings ({len(words)} words, explained var: {explained[0]:.2%}, {explained[1]:.2%})"
    plot_projection(coords, words, title, output_path, annotate_limit)
    print(f"PCA plot saved to: {output_path}")


def create_tsne_plot(word_matrix: np.ndarray, words: list[str], output_path: Path, annotate_limit: int, seed: int):
    if len(words) < 3:
        print("Skipping t-SNE: need at least 3 words.")
        return

    perplexity = min(30, max(5, len(words) // 3))
    perplexity = min(perplexity, len(words) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(word_matrix)
    title = f"t-SNE of embeddings ({len(words)} words, perplexity={perplexity})"
    plot_projection(coords, words, title, output_path, annotate_limit)
    print(f"t-SNE plot saved to: {output_path}")


def create_cosine_heatmap(
    word_matrix: np.ndarray,
    words: list[str],
    output_path: Path,
    heatmap_limit: int,
):
    if len(words) < 2:
        print("Skipping cosine heatmap: need at least 2 words.")
        return

    heatmap_words = words[:heatmap_limit]
    normalized = l2_normalize(word_matrix[: len(heatmap_words)])
    similarity = normalized @ normalized.T

    plt.figure(figsize=(max(8, len(heatmap_words) * 0.45), max(7, len(heatmap_words) * 0.45)))
    im = plt.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")
    plt.xticks(range(len(heatmap_words)), heatmap_words, rotation=90)
    plt.yticks(range(len(heatmap_words)), heatmap_words)
    plt.title(f"Cosine similarity heatmap ({len(heatmap_words)} words)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Cosine heatmap saved to: {output_path}")


def save_metadata(
    output_dir: Path,
    run_id: str,
    checkpoint_dir: Path,
    embedding_source: str,
    selected_words: list[str],
    missing_words: list[str],
):
    metadata = {
        "run_id": run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "embedding_source": embedding_source,
        "num_selected_words": len(selected_words),
        "selected_words": selected_words,
        "missing_words": missing_words,
    }

    metadata_path = output_dir / f"{embedding_source}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Visualization metadata saved to: {metadata_path}")


def main():
    args = parse_args()

    checkpoint_root = PROJECT_ROOT / "checkpoints"
    run_dir = checkpoint_root / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_config = load_run_config(run_dir)
    checkpoint_dir = resolve_checkpoint_dir(run_dir, args.checkpoint_subdir)
    embedding_matrix = load_embedding_matrix(checkpoint_dir, args.embedding_source)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"
    vocab_cache_dir = output_dir if args.output_dir else None
    vocab = load_or_build_vocab(run_dir, run_config, cache_dir=vocab_cache_dir)

    selected_words, missing_words = select_words(vocab, args.words, args.num_words)
    if missing_words:
        print(f"Words not found in vocab and skipped: {missing_words}")
    if len(selected_words) < 2:
        raise ValueError("Need at least 2 valid words to visualize.")
    print(f"Selected words for visualization ({len(selected_words)}): {selected_words}")

    word_matrix = build_word_matrix(embedding_matrix, vocab, selected_words)

    output_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{args.embedding_source}_{checkpoint_dir.name}"
    create_pca_plot(
        word_matrix=word_matrix,
        words=selected_words,
        output_path=output_dir / f"{file_prefix}_pca.png",
        annotate_limit=args.annotate_limit,
    )
    create_tsne_plot(
        word_matrix=word_matrix,
        words=selected_words,
        output_path=output_dir / f"{file_prefix}_tsne.png",
        annotate_limit=args.annotate_limit,
        seed=args.seed,
    )
    create_cosine_heatmap(
        word_matrix=word_matrix,
        words=selected_words,
        output_path=output_dir / f"{file_prefix}_cosine_heatmap.png",
        heatmap_limit=args.heatmap_limit,
    )
    save_metadata(
        output_dir=output_dir,
        run_id=args.run_id,
        checkpoint_dir=checkpoint_dir,
        embedding_source=args.embedding_source,
        selected_words=selected_words,
        missing_words=missing_words,
    )


if __name__ == "__main__":
    main()
