import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🦙",
    layout="centered"
)

# =========================================================
# Prompt Template
# =========================================================
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Respond clearly, concisely, and professionally."),
        ("human", "{question}")
    ]
)

# =========================================================
# LLM Response Generator
# =========================================================
def generate_response(question: str, model_name: str, temperature: float) -> str:
    """
    Generates a response from the selected Ollama model.

    Args:
        question (str): User input question
        model_name (str): Ollama model name
        temperature (float): Creativity level

    Returns:
        str: AI-generated response
    """
    try:
        llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )

        chain = PROMPT_TEMPLATE | llm | StrOutputParser()
        return chain.invoke({"question": question})

    except Exception as error:
        return f"⚠️ Error generating response: {error}"

# =========================================================
# UI Layout
# =========================================================
st.title("🤖 AI Chatbot")
st.subheader("An Open-Source LLM powered Q&A System")

# Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["mistral", "llama3", "gemma"]
)

temperature = st.sidebar.slider(
    "Response Creativity (Temperature)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

# =========================================================
# User Input Section
# =========================================================
st.markdown("### 💬 Ask your question below")

user_question = st.text_input(
    "Enter your question:",
    placeholder="Type your question here..."
)

# =========================================================
# Response Display
# =========================================================
if user_question:
    with st.spinner("Generating response..."):
        answer = generate_response(user_question, selected_model, temperature)

    st.markdown("### 🧠 AI Assistant")
    st.write(answer)
else:
    st.info("Please enter a question to continue.")
