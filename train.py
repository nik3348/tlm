import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from trm import TRM, TRMConfig


def main():
    # Load dataset and tokenizer
    print("Loading dataset and tokenizer...")
    dataset = load_dataset(
        "roneneldan/TinyStories", split="train[:10000]"
    )  # Using a subset for demonstration
    val_dataset = load_dataset(
        "roneneldan/TinyStories", split="train[10000:11000]"
    )  # Small validation set

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    config = TRMConfig(vocab_size=tokenizer.vocab_size)
    model = TRM(config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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
    best_val_loss = float("inf")
    eval_steps = 100
    model.train()
    for epoch in range(1):  # 1 epoch for demonstration
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)

            # For causality, we train to predict the next token.
            # In TRM model's logic, "question_ids" and "target_ids" have same shape.
            # We use shifted input ids for targets
            q_ids = input_ids[:, :-1]
            t_ids = input_ids[:, 1:]

            # The network expects question and target ids of same length, so we will not shift here but just match sizes.
            loss = model.train_step(q_ids, t_ids, optimizer)
            total_loss += loss

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")

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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print("Validation loss improved. Saving model...")
                    torch.save(model.state_dict(), "best_model.pt")

                model.train()


if __name__ == "__main__":
    main()
