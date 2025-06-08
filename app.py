import streamlit as st
from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage

from src.chatbot_logic import vector_store_from_chroma, create_chain


# Load environment variables from .env file
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

#tracing
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "squarebot_LCEL_Trace"

vector_store = vector_store_from_chroma(persist_directory="vectorstore")

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # k is the number of documents to retrieve


# Streamlit app
st.title("Squarebot 🟦")
st.write("Ask me anything!")

# initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# initializing the conversational history for LCEL chain
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# LCEL chain initialization
if "chain" not in st.session_state:
    st.session_state.chain = create_chain(retriever) # this creates the chain for the chatbot

# Display chat messages
# Loop through the messages in session state and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# User input
if user_input := st.chat_input("what can I help you with?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Thinking..."):
        # Debugging line to check the type of chain
        st.write(f"Type of chain: {type(st.session_state.chain)}")  
        
        # Process query from with chain
        response = st.session_state.chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.conversation_history
            })
        
        # LCEL Chain retruns a dictionary with "answer" and "context"
        answer = response["answer"]
        source_documents = response.get("context", []) # use .get() to avoid KeyError if "context" is not present

        # adding AI response to conversation history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Manually updating the conversation history for the next turn
        st.session_state.conversation_history.append(HumanMessage(content=user_input))
        st.session_state.conversation_history.append(AIMessage(content=answer))

        # Display bot message
        with st.chat_message("assistant"):
            st.markdown(answer)

            # Display source documents if available 
            if source_documents:
                st.subheader("Source Documents:")
                for i, doc in enumerate(source_documents):
                    st.write(f"**Document {i+1}**")
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.write(f"*Metadata: {doc.metadata}*")
                    st.write("---")

# clear chat history button in a sidebar
st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.update(
    messages=[],  # Clear the messages list
    conversation_history=[],  # Clear the conversation history
)) # this clears the chat history when the button is clicked






















