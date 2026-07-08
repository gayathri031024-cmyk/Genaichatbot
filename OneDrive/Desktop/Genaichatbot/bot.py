import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")
# =========================================================
def generate_response(question):
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error: {e}"

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
    response = generate_response(user_input)
    st.session_state.messages.append(("bot", response))

for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
