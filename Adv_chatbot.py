from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(page_title="Llama3 Chatbot", layout="centered")
st.title("🦙 LangChain Chatbot with Memory")

# Apply basic styling
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("system", "You are a helpful AI assistant. Please respond to the user's queries clearly and concisely.")]

# Temperature control
temperature = st.slider("🔧 Set Model Temperature", 0.0, 1.0, 0.7)

# Model and parser setup
llm = Ollama(model="deepseek-r1:1.5b", temperature=temperature)
output_parser = StrOutputParser()

# Clear Chat Button
if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = [("system", "You are a helpful AI assistant. Please respond to the user's queries clearly and concisely.")]
    st.rerun()

# Chat history display
# Chat history display with alignment
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"""
            <div style="text-align: right; background-color: #d1e7dd; padding: 10px; border-radius: 10px; margin: 10px 0;">
                <strong>You:</strong> {message}
            </div>
            """,
            unsafe_allow_html=True
        )
    elif role == "assistant":
        st.markdown(
            f"""
            <div style="text-align: left; background-color: #e2e3e5; padding: 10px; border-radius: 10px; margin: 10px 0;">
                <strong>Bot:</strong> {message}
            </div>
            """,
            unsafe_allow_html=True
        )


# 👉 Input field shown *after* chat history
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Type your next message here", key="user_input")
    submitted = st.form_submit_button("Send")

# If message submitted
if submitted and user_input:
    st.session_state.chat_history.append(("user", user_input))
    prompt = ChatPromptTemplate.from_messages(st.session_state.chat_history)
    chain = prompt | llm | output_parser

    with st.spinner("🤔 Thinking..."):
        try:
            response = chain.invoke({})
            st.session_state.chat_history.append(("assistant", response))
            st.rerun()  # Refresh to show response and move input down
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"Error: {e}"))
            st.rerun()
