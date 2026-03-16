import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Ensure project-root imports work even when executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.dataset import SkipGramBatchGenerator, generate_pairs
from src.data_prep.preprocess import build_vocab, encode_sentences, normalize_text, tokenize
from src.common.utils import load_vocab, load_wikitext_raw, save_vocab


SPECIAL_TOKENS = {"<PAD>", "<UNK>"}
DEFAULT_QUERY_CANDIDATES = [
    "king",
    "queen",
    "man",
    "woman",
    "child",
    "family",
    "city",
    "country",
    "world",
    "london",
    "paris",
    "government",
    "company",
    "music",
    "computer",
    "science",
    "history",
    "war",
    "play",
    "write",
    "build",
    "create",
    "support",
    "produce",
    "become",
    "large",
    "small",
    "major",
    "public",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SGNS embeddings on held-out splits.")
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
        help="Which embeddings to use for nearest-neighbor analysis",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["validation", "test"],
        choices=["train", "validation", "test"],
        help="Dataset splits used for held-out loss evaluation",
    )
    parser.add_argument(
        "--query-words",
        nargs="*",
        default=None,
        help="Specific words to use for nearest-neighbor evaluation",
    )
    parser.add_argument(
        "--num-query-words",
        type=int,
        default=8,
        help="How many fallback query words to use when --query-words is omitted",
    )
    parser.add_argument(
        "--top-k-neighbors",
        type=int,
        default=8,
        help="How many nearest neighbors to return per query word",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional cap on the number of sentences per evaluation split for faster iteration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional external directory for evaluation outputs and rebuilt vocab cache",
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

    print("Saved run-specific vocab not found. Rebuilding vocab from the training split in run_config.json.")
    raw_lines = load_wikitext_raw(run_config["split"])
    remove_stopwords = run_config.get("remove_stopwords", False)
    sentences = [tokenize(normalize_text(line), remove_stopwords=remove_stopwords) for line in raw_lines]
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


def load_embeddings(checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray]:
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
    return w_center, w_context


def select_neighbor_matrix(w_center: np.ndarray, w_context: np.ndarray, embedding_source: str) -> np.ndarray:
    if embedding_source == "center":
        return w_center
    if embedding_source == "context":
        return w_context
    return (w_center + w_context) / 2.0


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return -np.logaddexp(0.0, -x)


def compute_sgns_batch_loss(
    w_center: np.ndarray,
    w_context: np.ndarray,
    center_ids: np.ndarray,
    context_ids: np.ndarray,
    negative_ids: np.ndarray,
) -> float:
    v_c = w_center[center_ids]
    u_o = w_context[context_ids]
    u_neg = w_context[negative_ids]

    score_pos = np.sum(v_c * u_o, axis=1)
    score_neg = np.sum(u_neg * v_c[:, None, :], axis=2)

    loss_pos = -log_sigmoid(score_pos)
    loss_neg = -np.sum(log_sigmoid(-score_neg), axis=1)
    return float(np.mean(loss_pos + loss_neg))


def prepare_encoded_split(
    split: str,
    vocab: dict,
    run_config: dict,
    max_sentences: int | None = None,
) -> list[list[int]]:
    raw_lines = load_wikitext_raw(split)
    if max_sentences is not None:
        raw_lines = raw_lines[:max_sentences]
    remove_stopwords = run_config.get("remove_stopwords", False)
    sentences = [tokenize(normalize_text(line), remove_stopwords=remove_stopwords) for line in raw_lines]
    return encode_sentences(sentences, vocab)


def evaluate_split(
    split: str,
    vocab: dict,
    run_config: dict,
    w_center: np.ndarray,
    w_context: np.ndarray,
    max_sentences: int | None = None,
) -> dict:
    encoded_sentences = prepare_encoded_split(
        split,
        vocab,
        run_config,
        max_sentences=max_sentences,
    )
    pairs = generate_pairs(encoded_sentences, window_size=run_config["window_size"])
    batch_gen = SkipGramBatchGenerator(
        pairs=pairs,
        vocab=vocab,
        batch_size=run_config["batch_size"],
        num_negatives=run_config["num_negative_samples"],
        norm_factor=run_config["norm_factor"],
        seed=run_config["seed"],
        shuffle=False,
    )

    total_weighted_loss = 0.0
    total_examples = 0
    num_batches = 0

    for center_ids, context_ids, negative_ids in batch_gen:
        batch_loss = compute_sgns_batch_loss(
            w_center=w_center,
            w_context=w_context,
            center_ids=center_ids,
            context_ids=context_ids,
            negative_ids=negative_ids,
        )
        batch_size = int(center_ids.shape[0])
        total_weighted_loss += batch_loss * batch_size
        total_examples += batch_size
        num_batches += 1

    avg_loss = total_weighted_loss / total_examples if total_examples else None
    result = {
        "split": split,
        "num_sentences": len(encoded_sentences),
        "num_pairs": len(pairs),
        "num_batches": num_batches,
        "avg_loss": avg_loss,
        "max_sentences": max_sentences,
    }
    print(f"{split}: avg_loss={avg_loss}, pairs={len(pairs)}, batches={num_batches}")
    return result


def normalize_requested_word(raw_word: str) -> str | None:
    tokens = tokenize(normalize_text(raw_word))
    if len(tokens) != 1:
        return None
    return tokens[0]


def select_query_words(vocab: dict, requested_words: list[str] | None, num_query_words: int) -> tuple[list[str], list[str]]:
    word2id = vocab["word2id"]
    word_freq = vocab["word_freq"]
    missing = []

    if requested_words:
        selected = []
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

    selected = [word for word in DEFAULT_QUERY_CANDIDATES if word in word2id]
    if len(selected) < num_query_words:
        ranked_words = sorted(
            (word for word in word2id if word not in SPECIAL_TOKENS),
            key=lambda word: (-word_freq.get(word, 0), word),
        )
        for word in ranked_words:
            if word in selected:
                continue
            selected.append(word)
            if len(selected) >= num_query_words:
                break

    return selected[:num_query_words], missing


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def compute_nearest_neighbors(
    embedding_matrix: np.ndarray,
    vocab: dict,
    query_words: list[str],
    top_k: int,
) -> dict:
    normalized = l2_normalize(embedding_matrix)
    results = {}
    id2word = vocab["id2word"]

    for query_word in query_words:
        query_id = vocab["word2id"][query_word]
        similarities = normalized @ normalized[query_id]
        ranked_ids = np.argsort(-similarities)

        neighbors = []
        for candidate_id in ranked_ids:
            if int(candidate_id) == query_id:
                continue
            candidate_word = id2word[int(candidate_id)]
            if candidate_word in SPECIAL_TOKENS:
                continue
            neighbors.append(
                {
                    "word": candidate_word,
                    "score": float(similarities[int(candidate_id)]),
                }
            )
            if len(neighbors) >= top_k:
                break

        results[query_word] = neighbors
        neighbor_words = ", ".join(f"{item['word']} ({item['score']:.4f})" for item in neighbors)
        print(f"{query_word}: {neighbor_words}")

    return results


def save_json(output_path: Path, payload: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    checkpoint_root = PROJECT_ROOT / "checkpoints"
    run_dir = checkpoint_root / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_config = load_run_config(run_dir)
    checkpoint_dir = resolve_checkpoint_dir(run_dir, args.checkpoint_subdir)
    w_center, w_context = load_embeddings(checkpoint_dir)
    evaluation_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation"
    vocab_cache_dir = evaluation_dir if args.output_dir else None
    vocab = load_or_build_vocab(run_dir, run_config, cache_dir=vocab_cache_dir)

    evaluation_dir.mkdir(parents=True, exist_ok=True)

    split_results = [
        evaluate_split(
            split=split,
            vocab=vocab,
            run_config=run_config,
            w_center=w_center,
            w_context=w_context,
            max_sentences=args.max_sentences,
        )
        for split in args.eval_splits
    ]
    summary_payload = {
        "run_id": args.run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "eval_splits": split_results,
    }
    save_json(evaluation_dir / "heldout_loss_summary.json", summary_payload)

    neighbor_matrix = select_neighbor_matrix(w_center, w_context, args.embedding_source)
    query_words, missing_words = select_query_words(vocab, args.query_words, args.num_query_words)
    if missing_words:
        print(f"Query words not found in vocab and skipped: {missing_words}")
    if not query_words:
        raise ValueError("No valid query words available for nearest-neighbor evaluation.")

    neighbor_payload = {
        "run_id": args.run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "embedding_source": args.embedding_source,
        "query_words": query_words,
        "missing_query_words": missing_words,
        "neighbors": compute_nearest_neighbors(
            embedding_matrix=neighbor_matrix,
            vocab=vocab,
            query_words=query_words,
            top_k=args.top_k_neighbors,
        ),
    }
    save_json(evaluation_dir / f"nearest_neighbors_{args.embedding_source}.json", neighbor_payload)


if __name__ == "__main__":
    main()
