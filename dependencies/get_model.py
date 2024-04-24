import streamlit as st
import transformers
import torch
import peft
import huggingface_hub
import os


# @st.cache(allow_output_mutation=True)
@st.cache_resource()
def get_model(BASE_MODEL_ID: str, MODEL_DATA_SCIENCE_DIR: str, MODEL_SIVIREP_DIR: str):

    # Tokenizer base
    tokenizer_base = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer_base.pad_token = tokenizer_production.eos_token
    tokenizer_base.padding_side = "right"

    # Tokenizer produccion
    tokenizer_production = transformers.AutoTokenizer.from_pretrained(MODEL_SIVIREP_DIR)
    tokenizer_production.pad_token = tokenizer_production.eos_token
    tokenizer_production.padding_side = "right"

    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    huggingface_hub.login(token=os.environ.get("HFT"))

    # Model base
    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model_base.config.use_cache = False
    model_base.config.pretraining_tp = 1

    # Uniendo con R Data Science
    model_r_datascience = peft.PeftModel.from_pretrained(
        model_base, f"{MODEL_DATA_SCIENCE_DIR}"
    )
    model_r_datascience.eval()

    # Uniendo con Sivirep
    model_production = peft.PeftModel.from_pretrained(
        model_r_datascience, f"{MODEL_SIVIREP_DIR}"
    )
    model_production.eval()

    return tokenizer_production, model_production, tokenizer_base, model_base
