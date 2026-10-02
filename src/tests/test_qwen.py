from src.models.loader import load_qwen


def test_qwen_loads():

    model, tokenizer = load_qwen()

    prompt = "Return a JSON object containing a name."

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    print("\nInput tokens:")
    print(inputs["input_ids"])

    print("\nNumber of input tokens:")
    print(inputs["input_ids"].shape[-1])

    outputs = model.generate(
        **inputs,
        max_new_tokens=10
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nGenerated text:")
    print(generated_text)

    assert outputs.shape[-1] > inputs["input_ids"].shape[-1]