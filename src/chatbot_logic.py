import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st

# initializing llm and embeddings
llm = ChatGroq(model="Gemma2-9b-It") # updated model, better response than the original option. 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
# using the db vectors 
def vector_store_from_chroma(persist_directory="vectorstore"):
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

# Create a chain with the retriever and memory
def create_chain(retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer") # this is used to store the chat history
    # Custom prompt template
    custom_prompt = PromptTemplate(
        input_variables=["chat_history","question","context"],
        template=(
           "You are a helpful Q&A assistant focusing on Data Science and Machine Learning. "
            "Provide concise accurate answers based ONLY on the provided context.\n" # Added emphasis here
            "Do NOT include any internal thoughts, reasoning steps, or any tags like <think> or <answer>.\n" # Explicitly forbidden
            "Your response must be the direct answer and nothing else.\n"
            "If the answer is not in the context, say 'I don't know' or ask to rephrase the question.\n\n"
            "Use the following context to answer the question:\n\n {context}\n\n"
            "Question: {question}\n\nAnswer:" # Keep "Answer:" to prime the LLM for direct response
        )
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff", # "stuff" chain type is used for simple retrieval.
        retriever=retriever,
        memory=memory, # this is used to store the chat history
        return_source_documents=True, # this returns the source documents for the answer. False means it will not return the source document for the answer. instead it will return the answer only.
        combine_docs_chain_kwargs={"prompt": custom_prompt}, # this is used to set the prompt for the chain
    )
    return chain