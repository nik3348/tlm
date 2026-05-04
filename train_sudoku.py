import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from trm import TRM, TRMConfig

SUDOKU_VOCAB_SIZE = 10  # 0=padding/empty, 1-9=digits
PUZZLE_LEN = 81  # 9x9 board flattened


class SudokuDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        puzzle = row["question"]
        solution = row["answer"]

        q_ids = torch.tensor([int(c) for c in puzzle], dtype=torch.long)
        t_ids = torch.tensor([int(c) for c in solution], dtype=torch.long)
        return q_ids, t_ids


def save_checkpoint(model, optimizer, epoch, step, best_val_acc, checkpoint_path):
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return 0, 0, 0.0

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    torch.set_rng_state(checkpoint["rng_state"].cpu())
    if checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
        cuda_rng_state = [
            state.cpu() if isinstance(state, torch.Tensor) else state
            for state in checkpoint["cuda_rng_state"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state)

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    best_val_acc = checkpoint["best_val_acc"]
    print(f"Resumed from epoch {epoch}, step {step}, best val acc: {best_val_acc:.4f}")
    return epoch, step, best_val_acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_puzzles = 0
    correct_puzzles = 0
    total_cells = 0
    correct_cells = 0
    total_loss = 0

    for q_ids, t_ids in dataloader:
        q_ids = q_ids.to(device)
        t_ids = t_ids.to(device)

        loss = model.eval_step(q_ids, t_ids)
        total_loss += loss

        pred_ids = model.inference(q_ids)
        cell_match = pred_ids == t_ids

        total_puzzles += q_ids.size(0)
        correct_puzzles += (cell_match.all(dim=-1)).sum().item()
        total_cells += cell_match.numel()
        correct_cells += cell_match.sum().item()

    n = len(dataloader)
    return {
        "loss": total_loss / max(n, 1),
        "puzzle_acc": correct_puzzles / max(total_puzzles, 1),
        "cell_acc": correct_cells / max(total_cells, 1),
    }


def main():
    print("Loading dataset...")
    dataset = load_dataset("sapientinc/sudoku-extreme")
    train_data = SudokuDataset(dataset["train"])
    val_data = SudokuDataset(dataset["test"])

    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    config = TRMConfig(
        vocab_size=SUDOKU_VOCAB_SIZE,
        max_position_embeddings=PUZZLE_LEN,
        hidden_size=256,
        intermediate_size=682,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_latent_steps=6,
        t_deep_steps=3,
        n_supervision_steps=16,
    )
    model = TRM(config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100_000)

    checkpoint_dir = "checkpoints_sudoku"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    start_epoch, start_step, best_val_acc = load_checkpoint(
        model, optimizer, checkpoint_path, device
    )

    eval_steps = 5000
    checkpoint_steps = 2000
    num_epochs = 5
    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        epoch_puzzles = 0
        epoch_correct = 0

        for step, (q_ids, t_ids) in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                global_step += 1
                continue

            q_ids = q_ids.to(device)
            t_ids = t_ids.to(device)

            loss = model.train_step(q_ids, t_ids, optimizer)
            scheduler.step()

            # Compute cell accuracy for logging (quick argmax on last forward state)
            with torch.no_grad():
                x_embed = model.input_embedding(q_ids)
                z = torch.zeros_like(x_embed)
                y_embed = torch.zeros_like(x_embed)
                z = model.deep_recursion(x_embed, y_embed, z)
                logits = model.output_head(z)
                pred_ids = logits.argmax(dim=-1)
                cell_acc = (pred_ids == t_ids).float().mean().item()

            epoch_loss += loss
            epoch_puzzles += q_ids.size(0)
            epoch_correct += (pred_ids == t_ids).all(dim=-1).sum().item()
            global_step += 1

            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss:.4f} | "
                    f"Cell Acc: {cell_acc:.4f} | LR: {lr:.2e}"
                )

            if global_step % checkpoint_steps == 0:
                save_checkpoint(
                    model, optimizer, epoch, step, best_val_acc, checkpoint_path
                )

            if global_step % eval_steps == 0:
                val_metrics = evaluate(model, val_loader, device)
                print(
                    f"\n=== Validation @ step {global_step} ===\n"
                    f"  Loss: {val_metrics['loss']:.4f}\n"
                    f"  Puzzle Acc: {val_metrics['puzzle_acc']:.4f}\n"
                    f"  Cell Acc: {val_metrics['cell_acc']:.4f}\n"
                )

                if val_metrics["puzzle_acc"] > best_val_acc:
                    best_val_acc = val_metrics["puzzle_acc"]
                    best_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save(model.state_dict(), best_path)
                    print(f"  New best puzzle accuracy: {best_val_acc:.4f} — saved!")

                save_checkpoint(
                    model, optimizer, epoch, step, best_val_acc, checkpoint_path
                )
                model.train()

        train_puzzle_acc = epoch_correct / max(epoch_puzzles, 1)
        print(
            f"\nEpoch {epoch} complete — "
            f"Avg Loss: {epoch_loss / max(step + 1, 1):.4f} | "
            f"Train Puzzle Acc: {train_puzzle_acc:.4f}\n"
        )

    print("Training complete.")
    final_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    save_checkpoint(model, optimizer, num_epochs - 1, 0, best_val_acc, final_path)


if __name__ == "__main__":
    main()
