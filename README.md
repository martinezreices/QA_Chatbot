# Q&A Chatbot

This repository contains the code for a Q&A Chatbot designed to answer questions based on a custom knowledge base. The project leverages various tools from the LangChain ecosystem for document loading, splitting, embedding, and retrieval, with data processing specifically executed on Kaggle's GPU environment.

## Project Structure

The project is organized as follows:

├── data/                    # Placeholder for raw documents (not committed to Git)
├── notebooks/
│   └── q-a-chatbot.ipynb    # Jupyter notebook for data preparation and vectorization
├── src/
│   ├── chatbot_logic.py     # Core chatbot logic for the Q&A system
│   ├── data_processing.py   # Scripts for data handling (currently not being used)
│   └── vectorstore/         # Placeholder for the Chroma vector store (not committed to Git)
├── .env                     # Environment variables (excluded from Git)
├── .gitignore               # Specifies files/directories to ignore
├── app.py                   # Main application entry point (Streamlit application)
├── original_code.py         # Original or experimental code (kept for reference)
├── requirements.txt         # Python dependencies
└── README.md                # This README file

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/martinezreices/Q-A_Chatbot.git](https://github.com/martinezreices/Q-A_Chatbot.git)
    cd Q-A_Chatbot
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate.bat
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Key Configuration:**
    This project utilizes LangChain and Groq. You will need to obtain API keys for these services.
    * **For Kaggle Notebook Execution:** Load your API keys into Kaggle Secrets with the names:
        * `LANGCHAIN_API_KEY`
        * `GROQ_API_KEY`
    * **For Local Application Execution:** Set these as environment variables in your local development environment (e.g., in your `.env` file and load them using a library like `python-dotenv`).
        ```
        LANGCHAIN_API_KEY=your_langchain_api_key_here
        GROQ_API_KEY=your_groq_api_key_here
        ```

## Data Processing Pipeline (`q-a-chatbot.ipynb`)

The `notebooks/q-a-chatbot.ipynb` Jupyter notebook is the core of the data preparation and vectorization process. This notebook was designed to be executed on **Kaggle using their GPU environment** to efficiently handle the computational demands of embedding and vectorizing documents.

The pipeline in the notebook involves the following key steps:

1.  **Resource Loading:**
    * **PDF Document Loading:** Utilized `PyPDFLoader` from LangChain to ingest PDF documents.
    * **Web Page Loading:** Employed `RecursiveUrlLoader` from LangChain to load content from web pages containing numerous links, ensuring comprehensive data collection.
    * All loaded resources are then combined into a single, unified list of documents.

2.  **Document Splitting:**
    * The combined documents are split into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
    * Documents are split with a `chunk_size` of **1000 characters** and an `overlap` of **200 characters**. This strategy helps maintain context across chunks while preparing them for embedding.

3.  **Embedding and Vectorization:**
    * For generating document embeddings, the `all-MiniLM-L6-v2` model from HuggingFace was chosen. This model is accessed and integrated through LangChain's embeddings functionality.
    * The generated embeddings are then loaded into a **Chroma vector store**.

4.  **Retriever Setup:**
    * Finally, the initialized Chroma vector store is set up as a **retriever**. This retriever will be used by the chatbot to fetch relevant document chunks based on user queries, enabling context-aware responses.

## Usage

To execute the application file, ensure that you navigate to the project's root (`/Q&A Chatbot`) in your command line and then execute:

```bash
streamlit run app.py
```
