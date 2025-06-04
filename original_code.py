# the structure of app.py is as follows:
    # 1. Import necessary libraries and modules. ----  completed 
    # 2. Load environment variables from a .env file. ----  completed
    # 3. Initialize the ChatGroq model and embeddings. ----  completed
    # 4. Define functions to load documents, split them into chunks, and create a vector store. ----  completed
    # 5. Load documents and create a vector store. ----- working on this part 
    # 6. Create a retriever from the vector store. ----  completed
    # 7. Create a ChatGroq chain with the retriever. ----  completed
    # 8. Create a Streamlit app with a title and input field for user questions.
    # 9. When the user submits a question, display the answer and source documents.
    # 10. The app uses caching to optimize performance by storing the results of expensive computations.



import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import os

# Load environment variables from .env file
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

llm = ChatGroq(model="Qwen-Qwq-32b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # previously attempted Linq-AI-Research/Linq-Embed-Mistral, this model is too large for the PC

# we will set up the path for the files to reference for each question we ask the chatbot

# Load text documents
@st.cache_resource
def load_documents(file_paths):
    documents = []
    for path in file_paths:
        if not os.path.exists(path):
            st.error(f"File not found: {path}")
            continue
        # Load the text document using TextLoader
        loader = TextLoader(path)
        documents.extend(loader.load())
    return documents


#loading the pdf documents
@st.cache_resource
def load_pdf_documents(file_paths):
    documents = []
    for path in file_paths:
        if not os.path.exists(path):
            st.error(f"File not found: {path}")
            continue
        # Load the PDF document using PyPDFLoader
        loader = PyPDFLoader(path) # this loads the text documents, 
        documents.extend(loader.load()) # load the documents and add them to the list
    return documents

# Load web documents from a URL, the recursiveURL laoder will scan a max depth of 2 for files containing integration in the url path

@st.cache_resource
def load_web_documents(urls):
    documents = []
    for url in urls:
        try:
            if "docs/integrations/" in url:
                loader = RecursiveUrlLoader(url, max_depth=2)
            elif "docs/how-to" in url or "docs/tutorials/" in url:
                loader = RecursiveUrlLoader(url, max_depth=1)
            else:
                loader = RecursiveUrlLoader(url, max_depth=0)
            documents.extend(loader.load())
        
        except Exception as  e:
            st.error(f"Error loading URL: {url}: {e}")
            
    return documents
 # this will load the web documents from the url, the max depth is set to 2 to avoid loading too many documents


# Split documents into chunks
@st.cache_resource
def split_documents(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(_documents)
    return splits

# Create vector store
@st.cache_resource
def create_vector_store(_splits): 
    vector_store = Chroma.from_documents(_splits, embeddings, persist_directory="db")
    vector_store.persist()
    return vector_store

# documents to load

# this will load all pdfs in the directory
# pdf_documents_to_load = [os.path.join(r"C:\Users\nmart\_LangChainProjects\Projects\Q&A Chatbot\data", f) for f in os.listdir(r"C:\Users\nmart\_LangChainProjects\Projects\Q&A Chatbot\data") if f.endswith('.pdf')]
# this will load all text files in the directory
# txt_documents_to_load = [os.path.join(r"C:\Users\nmart\_LangChainProjects\Projects\Q&A Chatbot\data", f) for f in os.listdir(r"C:\Users\nmart\_LangChainProjects\Projects\Q&A Chatbot\data") if f.endswith('.txt')]


txt_documents_to_load = [] # this is the text document to load, the r is used to indicate that the string is a raw string, so that the backslashes are not treated as escape characters.
pdf_documents_to_load = [] # list of pdf documents to load

recurive_base_urls_to_load = [
    "https://python.langchain.com/docs/how_to/document_loader_web/",
    "https://python.langchain.com/docs/integrations/document_loaders/",
    "https://python.langchain.com/docs/integrations/vectorstores/",
    "https://python.langchain.com/docs/integrations/retrievers/",
    "https://python.langchain.com/docs/integrations/text_embedding/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]

direct_pages_to_load = [
    "https://python.langchain.com/docs/integrations/vectorstores/chroma",
    "https://python.langchain.com/docs/integrations/retrievers/contextual_compression",
]


web_urls_to_load = []
web_urls_to_load.extend(recurive_base_urls_to_load)
web_urls_to_load.extend(direct_pages_to_load) # this is the list of web urls to load, the r is used to indicate that the string is a raw string, so that the backslashes are not treated as escape characters.

for file in os.listdir(r"c:\Users\nmart\_LangChainProjects\Projects\Q&A Chatbot\data"):
    if file.endswith(".pdf"):
        pdf_documents_to_load.append(os.path.join(r"C:\Users\nmart\_LangChainProjects\Projects\Q&A Chatbot\data", file))
    if file.endswith(".txt"):
        txt_documents_to_load.append(os.path.join(r"C:\Users\nmart\_LangChainProjects\Projects\Q&A Chatbot\data", file))

# using the predefined functions to load the documents into a single list
all_documents = []
all_documents.extend(load_documents(txt_documents_to_load)) # this loads the text documents
all_documents.extend(load_pdf_documents(pdf_documents_to_load)) # this loads the pdf documents
all_documents.extend(load_web_documents(web_urls_to_load)) # this loads the web documents


# split the documents into chunks
splits = split_documents(all_documents)

# Create vector store
vector_store = create_vector_store(splits)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k is the number of documents to retrieve


# Create a chain with the retriever and memory
def create_chain():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer") # this is used to store the chat history
    # Custom prompt template
    custom_prompt = PromptTemplate(
        input_variables=["chat_history","question","context"],
        template=(
            "You are a helpful Q&A asistant focusing on Data Science and Machine Learning. Provide concise accurate answers"
            "Always provide only the direct answer Never include reasoning steps or any internal tags like <think>."
            "remove all internal tags prior to returning the answer."
            "If the answer is not in the context, say 'I don't know'.\n\n"
            "use the following context to answer the question:\n\n {context}\n\n" \
            "Question: {question}\n\nAnswer:"
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

# if we use output_key="answer" and return_source_documents=True, the response will be a dictionary with the answer and source documents.



# Streamlit app
st.title("Niki the Chatbot🐥")
st.write("Ask me anything!")

# initialize the chat history [ Temporary rempoval of the manual chat history]
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.session_state.chain = create_chain() # this creates the chain for the chatbot

# Display the chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# User input
if user_input := st.chat_input("what can I help you with?"):
    #Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Thinking..."):
        # Process query from with chain
        response = st.session_state.chain({"question": user_input})
        answer = response["answer"]
        # source_documents = response["source_documents"]

        # Display bot message
        with st.chat_message("assistant"):
            st.markdown(answer)



# # Display the chat history
# for message in st.session_state.chain.memory.chat_memory.messages: 
#     if message.type not in ["human", "ai"]:
#         continue
#     role = "user" if message.type == "human" else "assistant"
#     with st.chat_message(role):
#         st.markdown(message.content)


# clear chat history button in a sidebar
st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.chain.memory.clear()) # this clears the chat history when the button is clicked




















