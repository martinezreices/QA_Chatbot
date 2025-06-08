import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # website on runnables: https://python.langchain.com/docs/modules/runnables/overview
from langchain_core.output_parsers import StrOutputParser

# initializing llm and embeddings
llm = ChatGroq(model="Gemma2-9b-It") # updated model, better response than the original option. 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



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

    """
    following Langchain's ConversationalRetrievalChain 0.2.x
    Part 1 - Create a history-aware retrieval chain.
    """
    
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"), # this is used to store the chat history,
            ("user", "{input}"), # this is the user input
            ("user", "Given the above conversation, generate a standalone question to retrieve relevant documents.")
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        history_aware_retriever_prompt,
    )

    """
    Part 2 - Create a Document Combining Chain (QA Chain).
    This chain will take the retrieved documents and combine them into a single context for the LLM.
    """
    custom_qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            "You are a helpful Q&A assistant focusing on Data Science and Machine Learning. "
            "Provide concise accurate answers based ONLY on the provided context.\n" # Added emphasis here
            "Do NOT include any internal thoughts, reasoning steps, or any tags like <think> or <answer>.\n" # Explicitly forbidden
            "Your response must be the direct answer and nothing else.\n"
            "If the answer is not in the context, say 'I don't know' or ask to rephrase the question.\n\n"
            "Use the following context to answer the question:\n\n {context} \n\n"
            ),

            MessagesPlaceholder(variable_name="chat_history"), # this is used to store the chat history
            ("user", "{input}"), # this is the user input
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=custom_qa_prompt,
    )

    """
    Part 3 - Create the conversational retrieval chain. using the factory function `create_retrieval_chain`.
    This chain will combine the history-aware retriever and the document combining chain.
    """

    rag_chain = create_retrieval_chain(
        history_aware_retriever, # this is the history aware retriever
        combine_docs_chain, # this is the document combining chain
    )

    return rag_chain

    """    Part 4 - Create the final chain that will take the user input and chat history, pass it through the history-aware retriever, and then combine the documents to generate an answer.

    def _get_chat_history(input):
        return input.get("chat_history", [])
    
    final_chain = (
        RunnablePassthrough.assign(
            question= lambda x: x["question"],
            chat_history=_get_chat_history
        )
        | history_aware_retriever
        | combine_docs_chain
        | StrOutputParser()
    )

    # chain = ConversationalRetrievalChain(
    #     retriever=history_aware_retriever, # this is the history aware retriever
    #     combine_docs_chain=combine_docs_chain, # this is the document combining chain
    #     memory=memory, # this is used to store the chat history
    #     return_source_documents=True, # this returns the source documents for the answer. False means it will not return the source document for the answer. instead it will return the answer only.
    # )
    # return chain

    """


