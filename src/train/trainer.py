import math

from tqdm import tqdm

from src.data.dataset import build_validation_batch_generator


def compute_validation_loss(model, validation_pairs, vocab, hyperparams):
    if not validation_pairs:
        return None

    val_batch_gen = build_validation_batch_generator(validation_pairs, vocab, hyperparams)
    total_weighted_loss = 0.0
    total_examples = 0

    for center_id, context_id, negative_ids in val_batch_gen:
        loss, _ = model.forward(center_id, context_id, negative_ids)
        batch_size = int(center_id.shape[0])
        total_weighted_loss += float(loss) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        return None
    return total_weighted_loss / total_examples


def get_learning_rate(global_step, total_steps, hyperparams):
    peak_lr = hyperparams["learning_rate"]
    start_lr = hyperparams.get("learning_rate_start", peak_lr)
    min_lr = hyperparams.get("learning_rate_min", start_lr)
    warmup_ratio = hyperparams.get("learning_rate_warmup_ratio", 0.0)

    if total_steps <= 1:
        return peak_lr

    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0

    if warmup_steps and global_step <= warmup_steps:
        progress = global_step / warmup_steps
        return start_lr + (peak_lr - start_lr) * progress

    decay_steps = max(1, total_steps - warmup_steps)
    decay_progress = (global_step - warmup_steps) / decay_steps
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (peak_lr - min_lr) * cosine


def train_model(model, batch_gen, validation_pairs, vocab, hyperparams, checkpoint_every, latest_ckpt_dir):
    global_step = 0
    train_loss_records = []
    validation_loss_records = []
    validation_every = hyperparams.get("validation_every")
    total_steps = hyperparams["num_epochs"] * len(batch_gen)

    for epoch in range(hyperparams["num_epochs"]):
        pbar = tqdm(batch_gen, desc=f"Epoch {epoch + 1}", unit="batch", leave=True)

        for step, (center_id, context_id, negative_ids) in enumerate(pbar, start=1):
            global_step += 1

            loss, cache = model.forward(center_id, context_id, negative_ids)
            learning_rate = get_learning_rate(global_step, total_steps, hyperparams)
            model.backward(cache, learning_rate)
            model.update()

            loss_val = float(loss)
            train_loss_records.append(
                {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "step_in_epoch": step,
                    "loss": loss_val,
                }
            )

            if step % 10 == 0:
                pbar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{learning_rate:.5f}")

            if validation_pairs and validation_every and global_step % validation_every == 0:
                validation_loss = compute_validation_loss(model, validation_pairs, vocab, hyperparams)
                validation_loss_records.append(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "step_in_epoch": step,
                        "loss": validation_loss,
                    }
                )
                pbar.write(f"Validation loss at step {global_step}: {validation_loss:.6f}")

            if global_step % checkpoint_every == 0:
                model.save_embeddings(latest_ckpt_dir)
                pbar.write(f"Checkpoint updated at step {global_step}: {latest_ckpt_dir}")

    return train_loss_records, validation_loss_records, global_step
