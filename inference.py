import torch
from transformers import AutoTokenizer

from trm import TRM, TRMConfig


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    config = TRMConfig(vocab_size=tokenizer.vocab_size)
    print("Initializing model...")
    model = TRM(config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)

    try:
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        print("Loaded best_model.pt")
    except FileNotFoundError:
        print(
            "Warning: best_model.pt not found. Running inference with an untrained model."
        )

    model.eval()

    prompt = "Once upon a time in a small village,"
    print(f"Input Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Run inference
    with torch.no_grad():
        output_ids = model.inference(input_ids)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\nModel Output: {output_text}")


if __name__ == "__main__":
    main()
