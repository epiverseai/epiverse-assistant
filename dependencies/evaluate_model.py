import torch
import inspect


@torch.no_grad()
def evaluate_model(prompt: str, tokenizer, model):
    eval_prompt = inspect.cleandoc(f"""[INST] <<<SYS>>> You are an expert in any topic, please response the following  <<<SYS>>> {prompt} [/INST] Response:""")
    model_input = tokenizer.encode(eval_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    model_output = model.generate(
        input_ids=model_input,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(model_output[0], skip_special_tokens=False)
