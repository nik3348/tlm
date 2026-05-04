import os

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from trm import TRM, TRMConfig

MAZE_VOCAB = {"#": 0, " ": 1, "S": 2, "G": 3, "o": 4}
MAZE_VOCAB_SIZE = len(MAZE_VOCAB)
MAZE_LEN = 900  # 30x30


class MazeDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row["question"]
        answer = row["answer"]

        q_ids = torch.tensor([MAZE_VOCAB[c] for c in question], dtype=torch.long)
        t_ids = torch.tensor([MAZE_VOCAB[c] for c in answer], dtype=torch.long)
        return q_ids, t_ids


def save_checkpoint(model, optimizer, epoch, step, best_val_acc, path):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        path,
    )
    print(f"Saved {path}")


def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        return 0, 0, 0.0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    torch.set_rng_state(ckpt["rng_state"].cpu())
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [s.cpu() if isinstance(s, torch.Tensor) else s for s in ckpt["cuda_rng_state"]]
        )
    print(f"Resumed from epoch {ckpt['epoch']}, step {ckpt['step']}")
    return ckpt["epoch"], ckpt["step"], ckpt["best_val_acc"]


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total, correct_puzzle, correct_cell, total_cell, total_loss = 0, 0, 0, 0, 0.0

    for q_ids, t_ids in dataloader:
        q_ids, t_ids = q_ids.to(device), t_ids.to(device)

        loss = model.eval_step(q_ids, t_ids)
        total_loss += loss

        pred = model.inference(q_ids)
        match = pred == t_ids

        total += q_ids.size(0)
        correct_puzzle += match.all(dim=-1).sum().item()
        total_cell += match.numel()
        correct_cell += match.sum().item()

    n = len(dataloader)
    return {
        "loss": total_loss / max(n, 1),
        "puzzle_acc": correct_puzzle / max(total, 1),
        "cell_acc": correct_cell / max(total_cell, 1),
    }


def main():
    wandb.login()

    print("Loading maze dataset...")
    ds = load_dataset("sapientinc/maze-30x30-hard-1k")
    train_data = MazeDataset(ds["train"])
    val_data = MazeDataset(ds["test"])

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    trm_config = TRMConfig(
        vocab_size=MAZE_VOCAB_SIZE,
        max_position_embeddings=MAZE_LEN,
        hidden_size=256,
        intermediate_size=682,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_latent_steps=6,
        t_deep_steps=3,
        n_supervision_steps=16,
    )
    model = TRM(trm_config)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | Device: {device}")

    lr = 1e-4
    weight_decay = 0.01
    num_epochs = 100
    eval_steps = 500
    save_steps = 200

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50_000)

    run = wandb.init(
        project="trm-maze",
        config={
            "vocab_size": MAZE_VOCAB_SIZE,
            "seq_len": MAZE_LEN,
            "hidden_size": trm_config.hidden_size,
            "intermediate_size": trm_config.intermediate_size,
            "num_hidden_layers": trm_config.num_hidden_layers,
            "num_attention_heads": trm_config.num_attention_heads,
            "n_latent_steps": trm_config.n_latent_steps,
            "t_deep_steps": trm_config.t_deep_steps,
            "n_supervision_steps": trm_config.n_supervision_steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealing",
            "total_params": sum(p.numel() for p in model.parameters()),
        },
    )

    ckpt_dir = "checkpoints_maze"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")

    start_epoch, start_step, best_val_acc = load_checkpoint(model, optimizer, ckpt_path, device)
    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step, (q_ids, t_ids) in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                global_step += 1
                continue

            q_ids, t_ids = q_ids.to(device), t_ids.to(device)

            loss = model.train_step(q_ids, t_ids, optimizer)
            scheduler.step()

            with torch.no_grad():
                x_embed = model.input_embedding(q_ids)
                z = torch.zeros_like(x_embed)
                y_embed = torch.zeros_like(x_embed)
                z = model.deep_recursion(x_embed, y_embed, z)
                pred = model.output_head(z).argmax(dim=-1)
                cell_acc = (pred == t_ids).float().mean().item()
                puzzle_acc = (pred == t_ids).all(dim=-1).float().mean().item()

            epoch_loss += loss
            epoch_correct += (pred == t_ids).all(dim=-1).sum().item()
            epoch_total += q_ids.size(0)
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]

            if step % 20 == 0:
                print(
                    f"E{epoch} S{step} | loss {loss:.4f} | cell {cell_acc:.4f} | "
                    f"puzzle {puzzle_acc:.4f} | lr {current_lr:.2e}"
                )

            wandb.log(
                {
                    "train/loss": loss,
                    "train/cell_acc": cell_acc,
                    "train/puzzle_acc": puzzle_acc,
                    "train/lr": current_lr,
                    "epoch": epoch,
                },
                step=global_step,
            )

            if global_step % save_steps == 0:
                save_checkpoint(model, optimizer, epoch, step, best_val_acc, ckpt_path)

            if global_step % eval_steps == 0:
                val = evaluate(model, val_loader, device)
                print(
                    f"\n=== Val @ step {global_step} ===\n"
                    f"  loss: {val['loss']:.4f}\n"
                    f"  puzzle_acc: {val['puzzle_acc']:.4f}\n"
                    f"  cell_acc: {val['cell_acc']:.4f}\n"
                )

                wandb.log(
                    {
                        "val/loss": val["loss"],
                        "val/puzzle_acc": val["puzzle_acc"],
                        "val/cell_acc": val["cell_acc"],
                        "val/best_puzzle_acc": max(best_val_acc, val["puzzle_acc"]),
                    },
                    step=global_step,
                )

                if val["puzzle_acc"] > best_val_acc:
                    best_val_acc = val["puzzle_acc"]
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
                    print(f"  New best: {best_val_acc:.4f}")
                save_checkpoint(model, optimizer, epoch, step, best_val_acc, ckpt_path)
                model.train()

        avg_loss = epoch_loss / max(step + 1, 1)
        train_puzzle_acc = epoch_correct / max(epoch_total, 1)
        print(
            f"\nEpoch {epoch} done — avg_loss {avg_loss:.4f} | "
            f"train_puzzle_acc {train_puzzle_acc:.4f}\n"
        )

        wandb.log(
            {
                "epoch/avg_loss": avg_loss,
                "epoch/train_puzzle_acc": train_puzzle_acc,
            },
            step=global_step,
        )

    print("Training complete.")
    save_checkpoint(model, optimizer, num_epochs - 1, 0, best_val_acc, os.path.join(ckpt_dir, "final.pt"))
    wandb.finish()


if __name__ == "__main__":
    main()
