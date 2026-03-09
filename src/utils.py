import json
from pathlib import Path
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
