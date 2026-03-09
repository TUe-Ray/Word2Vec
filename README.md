# JetBrains Internship Task: Hallucination Detection Task #1

## Task desciption
Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the ideas behind word2vec, the code, gradient derivations, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.

# Setup

## Environment Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify virtual environment is active:**
   - You should see `(.venv)` prefix in your terminal prompt

---

# Datasets Preparation

## Download wikitext-2-raw-v1

Run the download script:
```bash
python download_dataset.py
```

This will:
- Create `data/` directory if not exists
- Download wikitext-2-raw-v1 dataset from Hugging Face
- Save to `data/wikitext-2-raw-v1/`


# Demo

# Training