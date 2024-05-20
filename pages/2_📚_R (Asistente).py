import dotenv

dotenv.load_dotenv()

import streamlit as st
import warnings
import dependencies
import constants


warnings.filterwarnings("ignore")

global question
global response


st.set_page_config(
    page_title="R Assistant",
    page_icon="🤗",
)

# Model to use to predict response
(
    tokenizer_production,
    model_production,
    tokenizer_base,
    model_base,
    vector_index,
    embed_model,
    llm,
) = dependencies.get_model.get_model(
    constants.BASE_MODEL_ID,
    constants.MODEL_DATA_SCIENCE_DIR,
    constants.MODEL_SIVIREP_DIR,
    constants.MODEL_EMBEDED_ID,
    constants.URLS_SIVIREP,
    constants.URLS_R_DATASCIENCE,
)

st.title("R Chatbot - Assistant 👋")

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¿Cómo puedo ayudarte?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres aprender de R?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        question = prompt

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            question = dependencies.translate.translate_es_en(question)
            response = dependencies.evaluate_model.evaluate_model_single(
                question, "R for Data Science", tokenizer_production, model_production
            )
            response = dependencies.get_answer.get_answer(response)
            response = dependencies.translate.translate_en_es(response)
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
