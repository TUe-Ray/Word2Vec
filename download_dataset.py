#!/usr/bin/env python
"""
Download wikitext-2-raw-v1 dataset from Hugging Face
"""
import os
from pathlib import Path
from datasets import load_dataset

# Create data directory if not exists
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

print("Downloading wikitext-2-raw-v1 dataset")


try:
    # Download and save dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Save to disk
    dataset_path = data_dir / "wikitext-2-raw-v1"
    dataset.save_to_disk(str(dataset_path))
    
    print(f"Dataset downloaded successfully to {dataset_path}")
    
    # Print dataset info
    print("\n" + "="*50)
    print("Dataset Info:")
    print("="*50)
    for split, data in dataset.items():
        print(f"{split:15s}: {len(data)} examples")
    print("="*50)
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    raise
