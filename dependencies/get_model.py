import streamlit as st
import transformers
import torch
import peft
import huggingface_hub
import llama_index
import llama_index.core.prompts.prompts
import llama_index.embeddings.langchain
import llama_index.llms.huggingface
import llama_index.readers.web
import os


# @st.cache(allow_output_mutation=True)
@st.cache_resource()
def get_model(BASE_MODEL_ID: str, MODEL_DATA_SCIENCE_DIR: str, MODEL_SIVIREP_DIR: str):

    huggingface_hub.login(token=os.environ.get("HFT"))

    # Tokenizer base
    tokenizer_base = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer_base.pad_token = tokenizer_base.eos_token
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


@st.cache_resource()
def get_rag_model(
    model, tokenizer, MODEL_EMBEDED_ID: str, urls_sivirep: str, urls_r_datascience: str
):

    # join all urls
    urls = urls_sivirep + urls_r_datascience

    # Web Scrap all documents
    documents = llama_index.readers.web.BeautifulSoupWebReader().load_data(urls)

    # System prompt for model
    system_prompt = "You are an expert assistant. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. Your goal is to answer questions as accurately as possible based on the instructions and context provided. You must type the code in markdown format between ```r and ```. You must finish your answer with [/RES]"
    query_wrapper_prompt = llama_index.core.PromptTemplate(
        "<|USER|>{query_str}<|ASSISTANT|>"
    )

    # Stopping Critiria
    stop_tokens = [
        "[/RES]",
        " [/RES]",
        "[INST]",
        " [INST]",
        "<<<SYS>>>",
        " <<<SYS>>>",
        "|>",
        "|",
    ]
    stopping_ids = [
        tokenizer.encode(stop_token, add_special_tokens=False)[0]
        for stop_token in stop_tokens
    ]

    # Configure llama-index LLM
    llm = llama_index.llms.huggingface.HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        stopping_ids=stopping_ids,
    )

    # Read and return embeddings from the model to do RAG
    embed_model = llama_index.embeddings.huggingface.HuggingFaceEmbedding(
        model_name=MODEL_EMBEDED_ID
    )

    llama_index.core.Settings.llm = llm
    llama_index.core.Settings.embed_model = embed_model

    return llama_index.core.VectorStoreIndex.from_documents(documents)
