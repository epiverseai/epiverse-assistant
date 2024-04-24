import torch
import inspect
import transformers


class EosListStoppingCriteria(transformers.StoppingCriteria):
    # def __init__(self, tokenizer_production, stop_token: str):
    #     self.eos_sequence = tokenizer_production.encode(
    #         stop_token, add_special_tokens=False
    #     )

    def __init__(self, tokenizer, stop_token: str):
        self.eos_sequence = tokenizer.encode(stop_token, add_special_tokens=False)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


@torch.no_grad()
def evaluate_model_single(
    instruction: str,
    category: str,
    tokenizer,
    model,
    max_new_tokens: int = 600,
):

    if category == "R for Data Science":
        eval_prompt = inspect.cleandoc(
            f"""[INST] You are an expert assistant in {category}. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. {instruction} [/INST] Response: [RES] """
        )
    elif category == "Epiverse":
        eval_prompt = inspect.cleandoc(
            f"""[INST] You are an expert assistant in R for {category}, a library for epidemiological data analysis. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. {instruction} [/INST] Response: [RES] """
        )
    else:
        eval_prompt = inspect.cleandoc(
            f"""[INST] Respond informatively and accurately to any question posed. This includes answering simple questions about mood, providing detailed explanations of technical or academic concepts, and offering step-by-step guides when necessary. Make sure to adjust the tone and level of detail of your response according to the complexity of the question and the context provided. If the question is ambiguous or lacks information, kindly request more details to provide a more precise answer. Your goal is to be helpful, educational, and clear in all your responses. {instruction} [/INST] Response: """
        )

    stopping_criteria = [EosListStoppingCriteria(tokenizer, "[/RES]")]
    model_input = tokenizer.encode(
        eval_prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")
    model_output = model.generate(
        input_ids=model_input,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    return tokenizer.decode(model_output[0], skip_special_tokens=False)


@torch.no_grad()
def evaluate_model_beam(
    instruction: str,
    category: str,
    tokenizer,
    model,
    max_new_tokens: int = 600,
):

    if category == "R for Data Science":
        eval_prompt = inspect.cleandoc(
            f"""[INST] You are an expert assistant in {category}. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. {instruction} [/INST] Response: [RES] """
        )
    elif category == "Epiverse":
        eval_prompt = inspect.cleandoc(
            f"""[INST] You are an expert assistant in R for {category}, a library for epidemiological data analysis. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. {instruction} [/INST] Response: [RES] """
        )
    else:
        eval_prompt = inspect.cleandoc(
            f"""[INST] Respond informatively and accurately to any question posed. This includes answering simple questions about mood, providing detailed explanations of technical or academic concepts, and offering step-by-step guides when necessary. Make sure to adjust the tone and level of detail of your response according to the complexity of the question and the context provided. If the question is ambiguous or lacks information, kindly request more details to provide a more precise answer. Your goal is to be helpful, educational, and clear in all your responses. [/INST] Response: """
        )

    stopping_criteria = [EosListStoppingCriteria(tokenizer, "[/RES]")]
    model_input = tokenizer.encode(
        eval_prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")
    # Aquí aplica beam search
    model_output = model.generate(
        input_ids=model_input,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=5,
        num_return_sequences=1,
        do_sample=True,
        temperature=1,
    )

    return tokenizer.decode(model_output[0], skip_special_tokens=False)
