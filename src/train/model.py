from pathlib import Path

import numpy as np


class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Common Word2Vec init: small near-zero input vectors and zero output vectors.
        bound = 0.5 / embedding_dim
        self.W_center = np.random.uniform(
            low=-bound,
            high=bound,
            size=(vocab_size, embedding_dim),
        ).astype(np.float32)
        self.W_context = np.random.uniform(
            low=-bound,
            high=bound,
            size=(vocab_size, embedding_dim),
        ).astype(np.float32)
        

    def load_embeddings(self, dir_path):
        dir_path = Path(dir_path)
        w_center_path = dir_path / "W_center.npy"
        w_context_path = dir_path / "W_context.npy"

        if not w_center_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {w_center_path}")
        if not w_context_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {w_context_path}")
        
        loaded_center = np.load(w_center_path)
        loaded_context = np.load(w_context_path)
        expected_shape = (self.vocab_size, self.embedding_dim)

        if loaded_center.shape != loaded_context.shape:
            raise ValueError(
                "Loaded embedding matrices must have the same shape, "
                f"got center={loaded_center.shape}, context={loaded_context.shape}"
            )
        if loaded_center.shape != expected_shape:
            raise ValueError(
                "Loaded embedding shape does not match the current model "
                f"configuration: expected {expected_shape}, got {loaded_center.shape}"
            )

        self.W_center = loaded_center
        print(f"Center embeddings loaded from {w_center_path} with shape {self.W_center.shape}")
        self.W_context = loaded_context
        print(f"Context embeddings loaded from {w_context_path} with shape {self.W_context.shape}")


    def forward(self, center_ids, context_ids, negative_ids):
        """
            center_ids  : shape (B,)
            context_ids : shape (B,)
            negative_ids: shape (B, K)
        """
        center_ids = np.asarray(center_ids, dtype=np.int64)
        context_ids = np.asarray(context_ids, dtype=np.int64)
        negative_ids = np.asarray(negative_ids, dtype=np.int64)

        v_c = self.W_center[center_ids]  # shape: (B, D)
        u_o = self.W_context[context_ids]  # shape: (B, D)
        u_neg = self.W_context[negative_ids]  # shape: (B, num_negatives, D)

        score_pos = np.sum(v_c * u_o, axis=1) # shape: (B,)
        score_neg = np.sum(u_neg * v_c[:, None, :], axis=2)  # shape: (B, K)
        sigmoid_pos = self.sigmoid(score_pos)  # shape: (B,)
        sigmoid_neg = self.sigmoid(score_neg)  # shape: (B, K)

        # Use stable log-sigmoid to avoid overflow/underflow in log and exp.
        loss_pos = -self.log_sigmoid(score_pos)                     # (B,)
        loss_neg = -np.sum(self.log_sigmoid(-score_neg), axis=1)    # (B,)
        loss = np.mean(loss_pos + loss_neg)   # scalar        # store cache for backward pass
        cache = {
            'center_ids': center_ids,
            'context_ids': context_ids,
            'negative_ids': negative_ids,
            'v_c': v_c,
            'u_o': u_o,
            'u_neg': u_neg,
            'sigmoid_pos': sigmoid_pos,
            'sigmoid_neg': sigmoid_neg,
            'batch_size': center_ids.shape[0],
        }
        return loss, cache
    
    def backward(self, cache, learning_rate):
        """
        Returns batch gradients:
            grad_center : (B, D)
            grad_context: (B, D)
            grad_neg    : (B, K, D)
        """
        v_c = cache['v_c']                   # (B, D)
        u_o = cache['u_o']                   # (B, D)
        u_neg = cache['u_neg']               # (B, K, D)
        sigmoid_pos = cache['sigmoid_pos']   # (B,)
        sigmoid_neg = cache['sigmoid_neg']   # (B, K)
        B = cache['batch_size']
        self.lr = learning_rate
        # calculate gradient for matrix
        # Positive pair gradients
        # dL/d(score_pos) = sigmoid(score_pos) - 1
        grad_score_pos = (sigmoid_pos - 1.0) / B          # (B,)

        # Negative pair gradients
        # dL/d(score_neg) = sigmoid(score_neg)
        grad_score_neg = sigmoid_neg / B                  # (B, K)


        # Gradient wrt center embeddings
        grad_center = grad_score_pos[:, None] * u_o + np.sum(
            grad_score_neg[:, :, None] * u_neg,
            axis=1
        )   # (B, D)

        # Gradient wrt positive context embeddings
        grad_context = grad_score_pos[:, None] * v_c      # (B, D)

        # Gradient wrt negative context embeddings
        grad_neg = grad_score_neg[:, :, None] * v_c[:, None, :]   # (B, K, D)
        self.grads = [grad_center, grad_context, grad_neg]
        self.cache = cache
        

    
    def update(self): 
        grad_center, grad_context, grad_neg = self.grads
        lr = self.lr
        center_ids = self.cache['center_ids']      # (B,)
        context_ids = self.cache['context_ids']    # (B,)
        negative_ids = self.cache['negative_ids']  # (B, K)

        np.add.at(self.W_center, center_ids, -lr * grad_center)
        np.add.at(self.W_context, context_ids, -lr * grad_context)
        

        # handle repeated negative ids correctly
        np.add.at(
                    self.W_context,
                    negative_ids.reshape(-1),
                    -lr * grad_neg.reshape(-1, self.embedding_dim)
                )
        

    def save_embeddings(self, dir_path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        w_center_path = dir_path / "W_center.npy"
        w_context_path = dir_path / "W_context.npy"

        np.save(w_center_path, self.W_center)
        np.save(w_context_path, self.W_context)

        print(f"Center embeddings saved to {w_center_path}")
        print(f"Context embeddings saved to {w_context_path}")

    def sigmoid(self, x):
        # Stable sigmoid implementation that avoids overflow in exp.
        x = np.asarray(x)
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out.astype(np.float32, copy=False)

    def log_sigmoid(self, x):
        # log(sigmoid(x)) = -logaddexp(0, -x), numerically stable.
        x = np.asarray(x, dtype=np.float64)
        return -np.logaddexp(0.0, -x)
