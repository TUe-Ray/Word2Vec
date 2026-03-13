from src.preprocess import normalize_text, tokenize, build_vocab, encode_sentences
from src.utils import load_wikitext_raw, save_vocab
from src.dataset import build_negative_samples, generate_pairs, build_negative_sampling_dist, SkipGramBatchGenerator

# preprocess the dataset and build the vocabulary
raw_lines = load_wikitext_raw('train')
sentences = [tokenize(normalize_text(line)) for line in raw_lines]
vocab = build_vocab(sentences, min_freq=2, max_vocab=50000)
encoded_sentences = encode_sentences(sentences, vocab)

# generate training pairs and negative samples
pairs = generate_pairs(encoded_sentences, window_size=2)
batch_gen = SkipGramBatchGenerator(
    pairs=pairs,
    vocab=vocab,
    batch_size=128,
    num_negatives=5,
    norm_factor=0.75,
    seed=42,
)
for center_id, context_id, negative_ids in batch_gen:
    print(f"Center ID: {center_id}, Context ID: {context_id}, Negative IDs: {negative_ids}")
    # train_step(center_id, context_id, negative_ids)  # Implement your training step here
    pass  # Just print the first batch for demonstration