import re
from collections import Counter
from typing import List, Dict

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

# Tokenize text
def tokenize(text: str) -> List[str]:
    """
    Tokenize the text:
    - Split by whitespace
    - Remove empty tokens
    """
    result = []
    for token in text.split():
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
    

