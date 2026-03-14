import numpy as np

def generate_pairs(sentences, window_size):
    """
    Input:  encoded_sentences = [[2, 3, 4, 5], [10, 11, 12], ...]
    Output: pairs = [(center_id, [context_id1, context_id2, ...]), ...]
    """
    pairs = [] 
    for sentence in sentences:
        for i, word in enumerate(sentence):
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    pairs.append((word, sentence[j]))
    return pairs


"""
<dbg> for offline testing of negative sampling distribution
"""
# def build_negative_samples(pairs, vocab, num_negatives, norm_factor, seed):
#     """
#     Input:  
#         pairs = [(center_id, context_id), ...] -> for excluding the positive context from negative sampling
#         vocab = {'word2id': {...}, 'id2word': {...}, 'word_freq': {...}, 'vocab_size': N}
#         num_negatives = K (number of negative samples per positive pair)
#         norm_factor = unigram exponent, default 0.75
#         seed = random seed for reproducibility
#     Output:
#         negative_samples = {center_id: [neg_id1, neg_id2, ...], ...} (pre-sampled negatives for each center word)
#     """

#     vocab_size = vocab['vocab_size']
#     word2id = vocab['word2id']
#     word_freq = vocab['word_freq']
#     random.seed(seed)

#     #align word_freq with vocab indices
#     freq_by_id = np.ones(vocab_size, dtype=np.float64)
#     for w, idx in word2id.items():
#         freq_by_id[idx] = float(word_freq.get(w, 1.0))

#     probs = np.power(freq_by_id, norm_factor)
#     probs = probs / probs.sum()

#     # remove <PAD> and <UNK> from negative sampling
#     pad_id = word2id["<PAD>"]
#     unk_id = word2id["<UNK>"]
#     probs[pad_id] = 0.0  # <PAD>
#     probs[unk_id] = 0.0  # <UNK>
#     probs = probs / probs.sum()  # re-normalize after zeroing out

#     # sample K negative samples for each center word
#     negative_samples = []
#     for center_id, context_id in pairs:
#         negs = []
#         while len(negs) < num_negatives:
#             neg_id = int(random.choice(vocab_size, p = probs))
#             if neg_id == context_id:
#                 continue
#             if neg_id in negs:
#                 continue

#             negs.append(neg_id)
#         negative_samples.append(negs)
    
#     return negative_samples

class SkipGramBatchGenerator:
    def __init__(self, pairs, vocab, batch_size = 128, num_negatives = 5, norm_factor = 0.75, seed = 42, shuffle = True):
        self.pairs = pairs
        self.vocab = vocab
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.norm_factor = norm_factor
        self.rng =np.random.default_rng(seed)    
        self.shuffle = shuffle
        
        # build negative sampling distribution
        self.probs = self._build_negative_sampling_distribution()

        self.indices = np.arange(len(self.pairs))
        self.ptr = 0

    def _build_negative_sampling_distribution(self):
        freq_by_id = np.ones(self.vocab['vocab_size'], dtype=np.float64)
        for w, idx in self.vocab['word2id'].items():
            freq_by_id[idx] = float(self.vocab['word_freq'].get(w, 1.0))

        probs = np.power(freq_by_id, self.norm_factor)
        probs = probs / probs.sum()

        # remove <PAD> and <UNK> from negative sampling
        pad_id = self.vocab['word2id']["<PAD>"]
        unk_id = self.vocab['word2id']["<UNK>"]
        probs[pad_id] = 0.0  # <PAD>
        probs[unk_id] = 0.0  # <UNK>
        probs = probs / probs.sum()  # re-normalize after zeroing out

        return probs
    
    def reset(self):
        self.ptr = 0
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.ptr >= len(self.pairs):
            raise StopIteration

        end = min(self.ptr + self.batch_size, len(self.pairs))
        batch_idx = self.indices[self.ptr:end]
        self.ptr = end

        centers = []
        pos_contexts = []
        neg_contexts = []

        for idx in batch_idx:
            center_id, context_id = self.pairs[idx]
            centers.append(center_id)
            pos_contexts.append(context_id)
            neg_contexts.append(self._sample_negatives(context_id))

        return (
            np.array(centers, dtype=np.int64),            # shape: (B,)
            np.array(pos_contexts, dtype=np.int64),       # shape: (B,)
            np.array(neg_contexts, dtype=np.int64),       # shape: (B, K)
        )
    
    def _sample_negatives(self, context_id):
        """
        Sample K negative ids with unigram^0.75 distribution (already in self.probs),
        excluding the positive context id and avoiding duplicates in one sample.
        """
        # valid ids: probability > 0 (PAD/UNK already zeroed out)
        candidate_ids = np.flatnonzero(self.probs > 0)

        # exclude true positive context
        candidate_ids = candidate_ids[candidate_ids != context_id]

        if candidate_ids.size == 0:
            raise ValueError(
                "No valid negative candidates available after filtering."
            )

        # re-normalize on the filtered candidate set
        candidate_probs = self.probs[candidate_ids]
        candidate_probs = candidate_probs / candidate_probs.sum()

        # SGNS commonly samples with replacement, so duplicates are allowed.
        neg_ids = self.rng.choice(
            candidate_ids,
            size=self.num_negatives,
            replace=True,
            p=candidate_probs,
        )
        return neg_ids.tolist()
    

    def __len__(self):
        return (len(self.pairs) + self.batch_size - 1) // self.batch_size