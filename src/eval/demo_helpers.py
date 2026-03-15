from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def resolve_checkpoint_paths(run_dir, checkpoint_subdir, w_center_path=None, w_context_path=None):
    if w_center_path is None:
        w_center_path = run_dir / checkpoint_subdir / "W_center.npy"
    else:
        w_center_path = Path(w_center_path)

    if w_context_path is None:
        candidate = run_dir / checkpoint_subdir / "W_context.npy"
        w_context_path = candidate if candidate.exists() else None
    else:
        w_context_path = Path(w_context_path)

    return w_center_path, w_context_path


def load_run_config(run_dir):
    cfg_path = Path(run_dir) / "run_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_numpy_embeddings(w_center_path, w_context_path=None, embedding_source="center"):
    W_center = np.load(w_center_path)
    W_context = np.load(w_context_path) if w_context_path is not None and Path(w_context_path).exists() else None

    if embedding_source == "center":
        W = W_center
    elif embedding_source == "context":
        if W_context is None:
            raise ValueError("EMBEDDING_SOURCE='context' requires W_context.npy")
        W = W_context
    elif embedding_source == "average":
        if W_context is None:
            raise ValueError("EMBEDDING_SOURCE='average' requires W_context.npy")
        if W_center.shape != W_context.shape:
            raise ValueError(f"Shape mismatch: {W_center.shape} vs {W_context.shape}")
        W = (W_center + W_context) / 2.0
    else:
        raise ValueError("EMBEDDING_SOURCE must be one of: center, context, average")

    return W_center, W_context, W


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _extract_word_mappings_from_vocab_obj(vocab_obj):
    if isinstance(vocab_obj, dict):
        if "word2id" in vocab_obj:
            word2id = vocab_obj["word2id"]
            id2word = vocab_obj.get("id2word")
        elif "token_to_id" in vocab_obj:
            word2id = vocab_obj["token_to_id"]
            id2word = vocab_obj.get("id_to_token")
        elif "stoi" in vocab_obj:
            word2id = vocab_obj["stoi"]
            id2word = vocab_obj.get("itos")
        elif all(isinstance(k, str) and isinstance(v, int) for k, v in vocab_obj.items()):
            word2id = vocab_obj
            id2word = None
        else:
            raise ValueError(f"Unsupported vocab dict keys: {list(vocab_obj.keys())[:10]}")
    else:
        raise ValueError(f"Unsupported vocab object type: {type(vocab_obj)}")

    if id2word is None:
        id2word = {i: w for w, i in word2id.items()}
    elif isinstance(id2word, list):
        id2word = {i: w for i, w in enumerate(id2word)}
    elif isinstance(id2word, dict):
        id2word = {int(k): v for k, v in id2word.items()}
    else:
        raise ValueError(f"Unsupported id2word type: {type(id2word)}")

    word2id = {str(k): int(v) for k, v in word2id.items()}
    return word2id, id2word


def load_vocab_from_file(vocab_path, mode):
    vocab_path = Path(vocab_path)
    if mode == "json":
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_obj = json.load(f)
    elif mode == "pkl":
        with open(vocab_path, "rb") as f:
            vocab_obj = pickle.load(f)
    elif mode == "npy":
        vocab_obj = np.load(vocab_path, allow_pickle=True).item()
    else:
        raise ValueError("mode must be json / pkl / npy")
    return _extract_word_mappings_from_vocab_obj(vocab_obj)


def rebuild_vocab_from_project(project_root, run_config=None):
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data.preprocess import normalize_text, tokenize, build_vocab
    from src.common.utils import load_wikitext_raw

    cfg = run_config or {}
    split = cfg.get("split", "train")
    min_freq = cfg.get("min_freq", 2)
    max_vocab = cfg.get("max_vocab_size", 50000)

    raw_lines = load_wikitext_raw(split)
    sentences = [tokenize(normalize_text(line)) for line in raw_lines]
    vocab_obj = build_vocab(sentences, min_freq=min_freq, max_vocab=max_vocab)
    return _extract_word_mappings_from_vocab_obj(vocab_obj)


def maybe_load_vocab(vocab_mode, vocab_path, project_root, run_config, use_run_config_for_vocab=True):
    if vocab_mode == "none":
        return None, None
    if vocab_mode in {"json", "npy", "pkl"}:
        if vocab_path is None:
            raise ValueError(f"VOCAB_MODE='{vocab_mode}' requires VOCAB_PATH")
        return load_vocab_from_file(vocab_path, vocab_mode)
    if vocab_mode == "rebuild":
        return rebuild_vocab_from_project(project_root, run_config if use_run_config_for_vocab else None)
    raise ValueError("VOCAB_MODE must be one of: rebuild, json, npy, pkl, none")


def cosine_similarity(vec_a, vec_b, eps=1e-12):
    denom = max(np.linalg.norm(vec_a) * np.linalg.norm(vec_b), eps)
    return float(np.dot(vec_a, vec_b) / denom)


def nearest_neighbors(word, embeddings, word2id, id2word, top_k=10, normalized_embeddings=None, exclude_self=True):
    if word not in word2id:
        raise KeyError(f"Word not in vocab: {word}")

    if normalized_embeddings is None:
        normalized_embeddings = l2_normalize_rows(embeddings)

    idx = word2id[word]
    sims = normalized_embeddings @ normalized_embeddings[idx]
    if exclude_self:
        sims[idx] = -np.inf
    top_ids = np.argsort(-sims)[:top_k]
    return [(id2word[int(i)], float(sims[int(i)])) for i in top_ids]


def analogy(a, b, c, embeddings, word2id, id2word, top_k=10, normalized_embeddings=None):
    for word in (a, b, c):
        if word not in word2id:
            raise KeyError(f"Word not in vocab: {word}")

    if normalized_embeddings is None:
        normalized_embeddings = l2_normalize_rows(embeddings)

    va = normalized_embeddings[word2id[a]]
    vb = normalized_embeddings[word2id[b]]
    vc = normalized_embeddings[word2id[c]]
    query = vb - va + vc
    query = query / max(np.linalg.norm(query), 1e-12)

    sims = normalized_embeddings @ query
    for w in (a, b, c):
        sims[word2id[w]] = -np.inf
    top_ids = np.argsort(-sims)[:top_k]
    return [(id2word[int(i)], float(sims[int(i)])) for i in top_ids]


def similarity_table(pairs, embeddings, word2id):
    rows = []
    for w1, w2 in pairs:
        if w1 in word2id and w2 in word2id:
            s = cosine_similarity(embeddings[word2id[w1]], embeddings[word2id[w2]])
            rows.append({"word_1": w1, "word_2": w2, "cosine_similarity": s})
        else:
            rows.append({"word_1": w1, "word_2": w2, "cosine_similarity": np.nan})
    return pd.DataFrame(rows)


def plot_loss_if_available(run_dir):
    loss_path = Path(run_dir) / "loss_history.csv"
    if not loss_path.exists():
        print(f"No loss_history.csv found at: {loss_path}")
        return None
    df = pd.read_csv(loss_path)
    if not {"global_step", "loss"}.issubset(df.columns):
        print("loss_history.csv exists but does not have the expected columns: global_step, loss")
        return df

    plt.figure(figsize=(8, 4))
    plt.plot(df["global_step"], df["loss"], linewidth=1.2)
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.show()
    return df


def collect_existing_words(words, word2id):
    return [w for w in words if w in word2id]


def plot_pca(words, embeddings, word2id, figsize=(10, 8)):
    words = collect_existing_words(words, word2id)
    if len(words) < 2:
        raise ValueError("Need at least 2 words for the PCA plot")

    vecs = np.stack([embeddings[word2id[w]] for w in words], axis=0)
    coords = PCA(n_components=2).fit_transform(vecs)

    plt.figure(figsize=figsize)
    plt.scatter(coords[:, 0], coords[:, 1], s=30)
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9, alpha=0.85)
    plt.title("PCA projection of selected word vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.show()


def plot_tsne(words, embeddings, word2id, max_words=250, random_state=42, figsize=(11, 9)):
    words = collect_existing_words(words, word2id)[:max_words]
    if len(words) < 5:
        raise ValueError("Need at least 5 words for the t-SNE plot")

    vecs = np.stack([embeddings[word2id[w]] for w in words], axis=0)
    perplexity = min(30, max(5, len(words) // 6))
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=random_state,
    ).fit_transform(vecs)

    plt.figure(figsize=figsize)
    plt.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.8)
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.8)
    plt.title(f"t-SNE projection ({len(words)} words)")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_similarity_heatmap(words, embeddings, word2id, figsize=(9, 7)):
    words = collect_existing_words(words, word2id)
    if len(words) < 2:
        raise ValueError("Need at least 2 words for the heatmap")

    vecs = np.stack([embeddings[word2id[w]] for w in words], axis=0)
    vecs = l2_normalize_rows(vecs)
    sim = vecs @ vecs.T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim, vmin=-1, vmax=1)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_yticklabels(words)
    ax.set_title("Cosine similarity heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def maybe_pick_default_words(id2word, limit=200):
    words = [
        id2word[i]
        for i in sorted(id2word.keys())[:limit]
        if isinstance(id2word[i], str) and id2word[i].isalpha()
    ]
    return words[:limit]
