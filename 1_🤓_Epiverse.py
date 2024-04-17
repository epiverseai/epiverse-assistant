import dependencies.evaluate_model
import streamlit as st
import warnings
import dependencies

warnings.filterwarnings("ignore")

global question
global response


st.set_page_config(
    page_title="Epiverse",
    page_icon="👋",
)

BASE_MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
tokenizer, model = dependencies.get_model.get_model()

st.title("Epiverse Chatbot - Sivirep")

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿Cómo puedo ayudarte con Epiverse - Sivirep?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres aprender de Sivirep?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        question = prompt

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            question = dependencies.translate.translate_es_en(question)
            response = dependencies.evaluate_model.evaluate_model(question, tokenizer, model)
            print(response)
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
