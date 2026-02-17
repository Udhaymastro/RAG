# RAG with MongoDB and Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** system consisting of two main components: data ingestion and querying. It allows you to build a local knowledge base from your own documents (PDFs and Text files) and query it using the **Llama 3.1** Large Language Model via **Ollama**.

## 🚀 Features

*   **Document Ingestion**: Automatically loads, splits, and embeds documents from a `data/` directory.
*   **Vector Storage**: Stores document embeddings and metadata in a local **MongoDB** database.
*   **Semantic Search**: Uses **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) to find the most relevant context for your queries.
*   **GenAI Integration**: Generates answers using **Ollama** (Llama 3.1) or **OpenRouter** (OpenAI compatible), grounded in your local data.
*   **CLI Interface**: Simple command-line tools for both ingestion and querying.

## 🛠️ Tech Stack

*   **Python 3.x**
*   **LangChain**: Framework for building LLM applications.
*   **MongoDB**: Document database for storing text chunks and vector embeddings.
*   **Ollama**: Local LLM runner (hosting Llama 3.1).
*   **HuggingFace**: For generating text embeddings.

## 📋 Prerequisites

1.  **MongoDB**: Ensure you have a local MongoDB instance running (default: `mongodb://localhost:27017`).
2.  **Ollama**: Install [Ollama](https://ollama.com/) and pull the required model:
    ```bash
    ollama pull llama3.1:8b
    ```

## 📦 Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Create a `.env` file in the root directory (optional, uses defaults otherwise):
    ```env
    MONGO_URI=mongodb://localhost:27017
    # OPENROUTER_API_KEY=your_key_here  # Only if using OpenRouter
    ```

3.  **Prepare Data**:
    Place your `.txt` or `.pdf` files inside the `data/` folder.

4.  **Ingest Datar**:
    Run the ingestion script to process documents and save embeddings to MongoDB:
    ```bash
    python ingest.py
    ```

5.  **Query the System**:
    Ask a question to your RAG system:
    ```bash
    python query.py --query "What is the main topic of the documents?"
    ```
    
    Or specify a different model provider:
    ```bash
    python query.py --query "Explain the architecture" --model openrouter
    ```
