# RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using MongoDB and Ollama.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Set up environment variables in a `.env` file (see `.env.example` if available, or source code).

3.  Run the ingestion script:
    ```bash
    python ingest.py
    ```

4.  Query the system:
    ```bash
    python query.py
    ```
