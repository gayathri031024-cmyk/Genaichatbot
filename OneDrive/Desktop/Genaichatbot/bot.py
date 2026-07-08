import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# =========================================================
# Prompt Template
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("human", "{question}")
    ]
)

# =========================================================
# LLM + Chain
# =========================================================
def generate_response(question, model_name="mistral", temperature=0.7):
    try:
        llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )

        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        return chain.invoke({"question": question})

    except Exception as e:
        return f"Error: {str(e)}"

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="Ollama Q&A Chatbot", page_icon="🦙")

st.title("AI Chatbot")

st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Select Open Source Model",
    ["mistral", "llama3", "gemma"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

st.write("Ask any question 👇")

user_input = st.text_input("You:")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    response = generate_response(user_input, model_name, temperature)
    st.session_state.messages.append(("bot", response))

for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
