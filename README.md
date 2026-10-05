# Agentic RAG Assistant (Mistral AI + Hugging Face)

An agentic Retrieval-Augmented Generation (RAG) assistant built using Streamlit, LangChain / LangGraph, **Mistral AI** (`mistral-large-latest`), and **Hugging Face** local embeddings (`sentence-transformers/all-MiniLM-L6-v2`).

## Features
- **Local Hugging Face Embeddings**: Converts text chunks to dense vectors using `sentence-transformers/all-MiniLM-L6-v2` locally on CPU without requiring paid embedding API calls.
- **Mistral AI Agent**: Powered by `ChatMistralAI` with tool invocation and conversational memory via LangGraph.
- **PDF Ingestion & Chunking**: Uploads and parses multiple PDF documents with `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.
- **Interactive Chat Interface**: Streamlit UI with conversation state, spinners, and document reset capability.

## Setup Instructions

1. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your Mistral API key:
   ```bash
   cp .env.example .env
   ```
   Add:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
