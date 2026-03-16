# SGNS Derivation Note

This note explains the implementation in plain language rather than formal paper style.

## 1. What the model is trying to learn

We use **Skip-gram with Negative Sampling (SGNS)**.

The idea is simple:

- pick one center word, for example `king`
- pick one real context word that appeared near it, for example `queen`
- ask the model to give this real pair a high score
- also sample a few fake context words, and ask the model to give those fake pairs a low score

Each word has two vectors:

- `W_center`: the vector when the word is used as the center word
- `W_context`: the vector when the word is used as a context word

For one pair, the score is just a dot product:

`score(center, context) = v_c dot u_o`

If two words should go together, we want this score to be large.
If they should not go together, we want this score to be small or negative.

## 2. The loss in human terms

For one positive pair `(center, true_context)` and `K` negative samples, the loss is:

`L = -log(sigmoid(v_c dot u_o)) - sum(log(sigmoid(-(v_c dot u_neg_k))))`

Read it like this:

- the first term rewards the model when the real context gets a high score
- the second term rewards the model when fake contexts get low scores

So the model is always doing two pushes at once:

- pull true pairs together
- push fake pairs apart

In the code, I average this loss over the batch.

## 3. Gradients Calculation

The positive score is:

`s_pos = v_c dot u_o`

Its loss term is:

`L_pos = -log(sigmoid(s_pos))`

The derivative of this term with respect to the score is:

`dL_pos / ds_pos = sigmoid(s_pos) - 1`

That means:

- if the model already gives a very high positive score, the gradient is small
- if the model gives a weak positive score, the gradient is larger and pushes harder

For one negative sample:

`s_neg = v_c dot u_neg`

`L_neg = -log(sigmoid(-s_neg))`

Its derivative with respect to the score is:

`dL_neg / ds_neg = sigmoid(s_neg)`

That means:

- if a fake pair gets a high score by mistake, the gradient becomes large
- the model then pushes those vectors apart

From those score-level gradients, the vector gradients are just dot-product gradients:

- gradient w.r.t. center vector = positive part + all negative parts
- gradient w.r.t. positive context vector = positive score gradient times center vector
- gradient w.r.t. negative context vector = negative score gradient times center vector

That is exactly what `src/train/model.py` computes.

## 4. Why `unigram^0.75` is used for negative sampling

If we sampled negatives exactly by raw word frequency:

- words like `the`, `of`, `and` would dominate too much
- the model would spend too much effort separating everything from a tiny set of ultra-common words

If we sampled negatives uniformly:

- very rare words would appear too often
- the fake examples would stop looking like realistic noise from natural text

The `frequency^0.75` rule is a compromise:

- common words are still sampled more often than rare words
- but not as aggressively as raw frequency

In practice this gives more useful negative examples and usually trains better than either extreme.

## 5. Why subsampling frequent words helps

Subsampling is the trick that randomly drops very frequent words during training.

This mainly targets words like:

- `the`
- `of`
- `and`
- punctuation-like tokens that appear everywhere

Why this helps:

- these tokens create a huge number of boring training pairs
- they dominate the gradient
- they make many words look similar just because they share generic contexts

So subsampling reduces the number of low-information examples and lets the model spend more capacity on informative word co-occurrences.

This usually improves:

- training speed
- embedding quality
- nearest-neighbor quality

## 6. Time and memory tradeoffs

This implementation is designed for clarity, not maximum throughput.

### Time tradeoffs

- Negative sampling is much cheaper than full softmax.
- Full softmax would score against the whole vocabulary for every training example.
- SGNS only scores one positive context and `K` negatives.

So per training pair, the rough cost becomes proportional to:

`O((K + 1) * embedding_dim)`

instead of:

`O(vocab_size * embedding_dim)`

That is the main reason word2vec became practical.

### Memory tradeoffs

- It stores two embedding matrices: `W_center` and `W_context`
- each has shape `(vocab_size, embedding_dim)`
- It also currently materializes all skip-gram pairs in memory before training

That means this code is easy to read, but it is not the most memory-efficient design.

If the dataset gets much larger, better options would be:

- generate pairs on the fly instead of storing all pairs
- precompute or cache negative-sampling helpers more aggressively
- stream data by sentence chunks instead of loading everything into one large pair list


