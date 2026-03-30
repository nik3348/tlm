import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from trm import TRM, TRMConfig


def save_checkpoint(
    model, optimizer, scheduler, epoch, step, best_val_loss, checkpoint_path
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint and return epoch, step, and best_val_loss."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0, float("inf")

    print(f"Loading checkpoint from {checkpoint_path}")
    # Load to CPU first to avoid doubling GPU memory usage
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore random states for reproducibility
    torch.set_rng_state(checkpoint["rng_state"].cpu())
    if checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
        cuda_rng_state = [
            state.cpu() if isinstance(state, torch.Tensor) else state
            for state in checkpoint["cuda_rng_state"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state)

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Resumed from epoch {epoch}, step {step}, best val loss: {best_val_loss:.4f}"
    )
    return epoch, step, best_val_loss


def main():
    # Load dataset and tokenizer
    print("Loading dataset and tokenizer...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    config = TRMConfig(vocab_size=tokenizer.vocab_size)
    model = TRM(config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Tokenizer text
    def encode(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized_dataset = dataset.map(encode, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    val_tokenized_dataset = val_dataset.map(
        encode, batched=True, remove_columns=["text"]
    )
    val_tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_tokenized_dataset, batch_size=8, shuffle=False)

    print("Starting training...")
    eval_steps = 2500
    checkpoint_steps = 1000
    num_epochs = 1  # 1 epoch for demonstration

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.05 * total_steps)

    # Warmup + Cosine Decay scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Checkpoint configuration
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/trm_experiment")

    # Try to resume from checkpoint
    start_epoch, start_step, best_val_loss = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path, device
    )

    model.train()

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            # Skip steps if resuming from checkpoint
            if epoch == start_epoch and step < start_step:
                continue
            input_ids = batch["input_ids"].to(device)

            # For causality, we train to predict the next token.
            # In TRM model's logic, "question_ids" and "target_ids" have same shape.
            # We use shifted input ids for targets
            q_ids = input_ids[:, :-1]
            t_ids = input_ids[:, 1:]

            # The network expects question and target ids of same length, so we will not shift here but just match sizes.
            loss = model.train_step(q_ids, t_ids, optimizer)
            scheduler.step()
            total_loss += loss

            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}, LR: {current_lr:.2e}"
                )

                # Log to TensorBoard
                writer.add_scalar("Loss/train", loss, step)
                writer.add_scalar("Learning_Rate", current_lr, step)

            # Save checkpoint periodically
            if step > 0 and step % checkpoint_steps == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    best_val_loss,
                    checkpoint_path,
                )

            if step > 0 and step % eval_steps == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_q_ids = val_input_ids[:, :-1]
                        val_t_ids = val_input_ids[:, 1:]
                        v_loss = model.eval_step(val_q_ids, val_t_ids)
                        val_loss += v_loss

                val_loss /= len(val_dataloader)
                print(f"Validation Loss: {val_loss:.4f}")

                # Log eval metrics
                writer.add_scalar("Loss/val", val_loss, step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print("Validation loss improved. Saving best model...")
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    # Also save checkpoint with best validation loss
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        best_val_loss,
                        checkpoint_path,
                    )

                model.train()

    # Save final checkpoint at end of training
    print("Training complete. Saving final checkpoint...")
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    save_checkpoint(
        model, optimizer, scheduler, epoch, step, best_val_loss, final_checkpoint_path
    )

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
