# Task 4: Context-Aware Chatbot using LangChain & RAG

## Project Overview

Developed a Retrieval-Augmented Generation (RAG) chatbot capable of maintaining conversational context and retrieving information from a custom document corpus. The system utilizes a modular AI stack to provide accurate, data-backed responses.

## Key Features

- **Vector Retrieval**: Efficient document search using FAISS (Facebook AI Similarity Search)
- **Contextual Memory**: Built-in conversation history tracking to handle follow-up questions
- **Hugging Face Integration**: Uses Llama-3-8B-Instruct for high-quality natural language generation
- **Streamlit UI**: A clean, web-based interface for real-time interaction

## Tech Stack

- **Framework**: LangChain / LangChain-Classic
- **LLM**: Meta-Llama-3-8B-Instruct (via Hugging Face API)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Deployment**: Streamlit

## Implementation Steps

### 1. Data Ingestion & Vectorization

- Loaded PDF documents using PyPDFLoader
- Fragmented text into 1000-character chunks with RecursiveCharacterTextSplitter
- Generated embeddings and persisted the data locally in a `faiss_index` folder

### 2. Conversational Logic

- Configured ConversationBufferMemory to store chat history
- Implemented ConversationalRetrievalChain to link the LLM, the vector store, and the memory buffer

### 3. Frontend Deployment

- Built the user interface using Streamlit
- Managed session states to ensure the AI "remembers" the chat history across multiple user inputs

## How to Run

### Install Dependencies

```bash
pip install langchain-huggingface langchain-community langchain-classic faiss-cpu pypdf streamlit
```

### Generate Index

Run the ingestion script to create the `faiss_index` folder.

### Launch App

```bash
streamlit run chatbot_app.py
```
