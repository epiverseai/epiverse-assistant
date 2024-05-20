import torch
import inspect
import transformers

import llama_index
import llama_index.core.prompts.prompts
import llama_index.embeddings.langchain
import llama_index.llms.huggingface
import llama_index.readers.web


class EosListStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens: list[str]):
        self.eos_sequence = [
            tokenizer.encode(stop_token, add_special_tokens=False)
            for stop_token in stop_tokens
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # print("---")
        # for s in self.eos_sequence:
        #     print(s, input_ids[0][-len(s):].tolist())
        # print("---\n\n")
        return any([s == input_ids[0][-len(s) :].tolist() for s in self.eos_sequence])


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
            f"""[INST] <<<SYS>>> You are an expert assistant in {category}. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. <<</SYS>>> {instruction} [/INST] Response: [RES] """
        )
    elif category == "Epiverse":
        eval_prompt = inspect.cleandoc(
            f"""[INST] <<<SYS>>> You are an expert assistant in R for {category}, a library for epidemiological data analysis. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. <<</SYS>>> {instruction} [/INST] Response: [RES] """
        )
    else:
        eval_prompt = inspect.cleandoc(
            f"""[INST] <<<SYS>>> Respond informatively and accurately to any question posed. This includes answering simple questions about mood, providing detailed explanations of technical or academic concepts, and offering step-by-step guides when necessary. Make sure to adjust the tone and level of detail of your response according to the complexity of the question and the context provided. If the question is ambiguous or lacks information, kindly request more details to provide a more precise answer. Your goal is to be helpful, educational, and clear in all your responses. <<</SYS>>> {instruction} [/INST] Response: """
        )

    stopping_criteria = [EosListStoppingCriteria(tokenizer, ["[/RES]", " [/RES]"])]
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
            f"""[INST] <<<SYS>>> You are an expert assistant in {category}. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. <<</SYS>>> {instruction} [/INST] Response: [RES] """
        )
    elif category == "Epiverse":
        eval_prompt = inspect.cleandoc(
            f"""[INST] <<<SYS>>> You are an expert assistant in R for {category}, a library for epidemiological data analysis. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. <<</SYS>>> {instruction} [/INST] Response: [RES] """
        )
    else:
        eval_prompt = inspect.cleandoc(
            f"""[INST] <<<SYS>>> Respond informatively and accurately to any question posed. This includes answering simple questions about mood, providing detailed explanations of technical or academic concepts, and offering step-by-step guides when necessary. Make sure to adjust the tone and level of detail of your response according to the complexity of the question and the context provided. If the question is ambiguous or lacks information, kindly request more details to provide a more precise answer. Your goal is to be helpful, educational, and clear in all your responses. <<</SYS>>> {instruction} [/INST] Response: """
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


@torch.no_grad()
def evaluate_model_rag(instruction: str, embed_model, documents, llm):
    llama_index.core.Settings.llm = llm
    llama_index.core.Settings.embed_model = embed_model
    vector_index = llama_index.core.VectorStoreIndex.from_documents(documents)
    query_engine = vector_index.as_query_engine(response_mode="compact")
    response = query_engine.query(instruction)
    return response.response
