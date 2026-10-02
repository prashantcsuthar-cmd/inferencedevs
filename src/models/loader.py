import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_qwen():
    """
    Load the Qwen2.5-0.5B-Instruct model and tokenizer.

    ICE currently targets CPU-first local execution.
    """

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model.eval()

    print("Model loaded successfully.")
    print("Device: CPU")

    return model, tokenizer