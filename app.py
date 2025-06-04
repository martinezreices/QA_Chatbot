import streamlit as st
from dotenv import load_dotenv
import os

from src.chatbot_logic import vector_store_from_chroma, create_chain


# Load environment variables from .env file
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

vector_store = vector_store_from_chroma(persist_directory="vectorstore")

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # k is the number of documents to retrieve


# Streamlit app
st.title("Squarebot 🟦")
st.write("Ask me anything!")


if "chain" not in st.session_state:
    st.session_state.chain = create_chain(retriever) # this creates the chain for the chatbot


# User input
if user_input := st.chat_input("what can I help you with?"):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Thinking..."):
        # Process query from with chain
        response = st.session_state.chain({"question": user_input})
        answer = response["answer"]

        # Display bot message
        with st.chat_message("assistant"):
            st.markdown(answer)


# clear chat history button in a sidebar
st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.chain.memory.clear()) # this clears the chat history when the button is clicked






















