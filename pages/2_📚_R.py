import streamlit as st
import inspect
import transformers
import torch
import re
import warnings
import sys
import os
import dependencies.translate

warnings.filterwarnings("ignore")

global question
global response


# @st.cache(allow_output_mutation=True)
@st.cache_resource()
def get_model():
    BASE_MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"

    tokenizer_production = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer_production.pad_token = tokenizer_production.eos_token
    tokenizer_production.padding_side = "right"

    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_production = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model_production.config.use_cache = False
    model_production.config.pretraining_tp = 1
    return tokenizer_production, model_production


@torch.no_grad()
def evaluate_model(prompt: str):
    eval_prompt = inspect.cleandoc(
        f"""[INST] <<<SYS>>> You are an expert in any topic, please response the following  <<<SYS>>> {prompt} [/INST] Response:"""
    )
    model_input = tokenizer_production.encode(
        eval_prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")
    model_output = model_production.generate(
        input_ids=model_input,
        max_new_tokens=200,
        pad_token_id=tokenizer_production.eos_token_id,
        eos_token_id=tokenizer_production.eos_token_id,
    )
    return tokenizer_production.decode(model_output[0], skip_special_tokens=False)


def only_answer(response: str):
    result = re.search(r"Response: (.*)", response)
    if result:
        final_response = result.group(1)
        return final_response
    else:
        return response


st.set_page_config(
    page_title="R",
    page_icon="🤗",
)

# tokenizer_production, model_production = get_model()

# st.title("R Chatbot - Assistant")

# # Store LLM generated responses
# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "¿Cómo puedo ayudarte con R?"}]

# # Display or clear chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("¿Qué quieres aprender de R?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#         question = prompt

# # Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             question = dependencies.translate.translate_es_en(question)
#             response = evaluate_model(question)
#             response = only_answer(response)
#             response = dependencies.translate.translate_en_es(response)
#             placeholder = st.empty()
#             full_response = ""
#             for item in response:
#                 full_response += item
#                 placeholder.markdown(full_response)
#             placeholder.markdown(full_response)
#     message = {"role": "assistant", "content": full_response}
#     st.session_state.messages.append(message)
