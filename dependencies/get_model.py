import streamlit as st
import transformers
import torch


# @st.cache(allow_output_mutation=True)
@st.cache_resource()
def get_model(BASE_MODEL_ID: str):

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return tokenizer, model
