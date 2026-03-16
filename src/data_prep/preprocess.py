import re
from collections import Counter
from typing import List, Dict

import numpy as np

from .dataset import generate_pairs
from ..common.utils import load_wikitext_raw


MIN_TOKEN_ALNUM_LENGTH = 3
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "both", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his", "however",
    "i", "if", "in", "into", "is", "it", "its", "itself", "me", "my", "not", "of", "on",
    "or", "our", "ours", "she", "so", "than", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "which", "who", "will",
    "with", "you", "your", "also", "after", "before", "again", "more", "same", "can",
    "too", "all", "many", "while", "still", "such", "other", "around", "only", "during",
    "part", "between", "over", "under", "through", "about", "when", "where", "would",
    "made", "then", "some", "new", "old", "along", "just", "include", "including",
    "called",
}


# Normalize text
def normalize_text(text: str) -> str:
    """
    Clean the text:
    - Convert to lowercase
    - Remove special characters (keep only letters, numbers, spaces, and basic punctuation)
    - Remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?;]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def should_keep_token(token: str, min_alnum_length: int = MIN_TOKEN_ALNUM_LENGTH) -> bool:
    """
    Filter out low-signal tokens before vocab building and pair generation.

    Removes:
    - pure punctuation
    - tokens without alphabetic characters
    - digit-dominant tokens
    - very short tokens
    """
    alpha_count = sum(char.isalpha() for char in token)
    if alpha_count == 0:
        return False

    digit_count = sum(char.isdigit() for char in token)
    if digit_count > alpha_count:
        return False

    alnum_count = alpha_count + digit_count
    if alnum_count < min_alnum_length:
        return False

    return True


# Tokenize text
def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Tokenize the text:
    - Split by whitespace
    - Remove low-signal tokens
    """
    result = []
    for token in text.split():
        if not token or not should_keep_token(token):
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        if token:
            result.append(token)
    return result
    #return [token for token in text.split() if token]

# Build vocabulary
def build_vocab(sentences: List[List[str]], min_freq: int = 5, max_vocab: int = 50000) -> Dict:
    """
    Build the vocabulary:
    - Count token frequencies
    - Filter tokens based on min_freq and max_vocab
    - Create word2id / id2word mappings (add <PAD>, <UNK>)
    """
    # Count frequency of all tokens
    counter = Counter(token for sentence in sentences for token in sentence)
    
    # Filter tokens by minimum frequency and vocabulary size limit
    filtered_tokens = counter.most_common(max_vocab)
    filtered_tokens = [(word, freq) for word, freq in filtered_tokens if freq >= min_freq]
    
    # Create word to ID mapping with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(filtered_tokens, start=2):
        vocab[word] = idx
    
    # Create reverse mapping from ID to word
    id2word = {idx: word for word, idx in vocab.items()}
    
    return {
        'word2id': vocab,
        'id2word': id2word,
        'word_freq': dict(counter),
        'vocab_size': len(vocab)
    }


def subsample_frequent_words(
    sentences: List[List[str]],
    vocab: Dict,
    threshold: float = 1e-5,
    seed: int = 42,
) -> tuple[List[List[str]], Dict]:
    """
    Subsample frequent words using the Word2Vec keep-probability heuristic.

    Args:
        sentences: Tokenized sentences.
        vocab: Vocabulary dictionary returned by build_vocab.
        threshold: Subsampling threshold t. Set <= 0 to disable subsampling.
        seed: Random seed for reproducibility.

    Returns:
        (subsampled_sentences, stats)
    """
    total_tokens = sum(len(sentence) for sentence in sentences)
    if total_tokens == 0 or threshold <= 0:
        return sentences, {
            "subsample_threshold": threshold,
            "tokens_before": total_tokens,
            "tokens_after": total_tokens,
            "dropped_tokens": 0,
            "drop_ratio": 0.0,
            "sentences_before": len(sentences),
            "sentences_after": len(sentences),
        }

    rng = np.random.default_rng(seed)
    word_freq = vocab["word_freq"]
    word2id = vocab["word2id"]

    subsampled_sentences = []
    kept_tokens = 0

    for sentence in sentences:
        filtered_sentence = []
        for token in sentence:
            if token not in word2id:
                filtered_sentence.append(token)
                kept_tokens += 1
                continue

            freq_ratio = word_freq.get(token, 0) / total_tokens
            if freq_ratio <= 0:
                filtered_sentence.append(token)
                kept_tokens += 1
                continue

            keep_prob = min(1.0, (np.sqrt(freq_ratio / threshold) + 1.0) * (threshold / freq_ratio))
            if rng.random() < keep_prob:
                filtered_sentence.append(token)
                kept_tokens += 1

        if filtered_sentence:
            subsampled_sentences.append(filtered_sentence)

    dropped_tokens = total_tokens - kept_tokens
    stats = {
        "subsample_threshold": threshold,
        "tokens_before": total_tokens,
        "tokens_after": kept_tokens,
        "dropped_tokens": dropped_tokens,
        "drop_ratio": (dropped_tokens / total_tokens) if total_tokens else 0.0,
        "sentences_before": len(sentences),
        "sentences_after": len(subsampled_sentences),
    }
    return subsampled_sentences, stats

# Encode sentences
def encode_sentences(sentences: List[List[str]], vocab: Dict, unk_token: str = '<UNK>') -> List[List[int]]:
    """
    Encode sentences into ID sequences:
    - Convert each token to its corresponding ID
    - Tokens not in the vocabulary are mapped to <UNK> ID
    """
    word2id = vocab['word2id']
    unk_id = word2id[unk_token] # =1
    encoded_sentences = []

    for sentence in sentences:
        encoded_sentence = []
        for token in sentence:
            if token not in word2id:
                encoded_sentence.append(unk_id)
            else:
                encoded_sentence.append(word2id[token])
        encoded_sentences.append(encoded_sentence)
    return encoded_sentences


def prepare_training_data(hyperparams: Dict) -> tuple[Dict, list[tuple[int, int]]]:
    raw_lines = load_wikitext_raw(hyperparams["split"])
    remove_stopwords = hyperparams.get("remove_stopwords", False)
    sentences = [tokenize(normalize_text(line), remove_stopwords=remove_stopwords) for line in raw_lines]
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


def prepare_validation_data(vocab: Dict, hyperparams: Dict) -> list[tuple[int, int]] | None:
    validation_split = hyperparams.get("validation_split")
    if not validation_split:
        return None

    raw_lines = load_wikitext_raw(validation_split)
    max_sentences = hyperparams.get("validation_max_sentences")
    if max_sentences is not None:
        raw_lines = raw_lines[:max_sentences]

    remove_stopwords = hyperparams.get("remove_stopwords", False)
    sentences = [tokenize(normalize_text(line), remove_stopwords=remove_stopwords) for line in raw_lines]
    encoded_sentences = encode_sentences(sentences, vocab)
    pairs = generate_pairs(encoded_sentences, window_size=hyperparams["window_size"])
    print(
        f"Validation data prepared from split='{validation_split}': "
        f"{len(encoded_sentences)} sentences, {len(pairs)} pairs"
    )
    return pairs
    
