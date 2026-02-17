import os
import argparse
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# from ollama_llm import ask_ollama

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "rag_db"
COLLECTION_NAME = "Rag_doc"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Ensure this is in .env if using OpenRouter

def get_db_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def vector_search(query, collection, embeddings_model, k=3):
    query_embedding = embeddings_model.embed_query(query)
    
    # Retrieve all documents (fetching only necessary fields)
    # Optimization: In production, use Atlas Vector Search or a dedicated vector DB.
    # For local MongoDB without vector index, we fetch ALL and compute basic similarity.
    cursor = collection.find({}, {"text": 1, "embedding": 1})
    
    results = []
    for doc in cursor:
        score = cosine_similarity(query_embedding, doc['embedding'])
        results.append((score, doc['text']))
    
    # Sort by score desc and take top k
    results.sort(key=lambda x: x[0], reverse=True)
    return [text for score, text in results[:k]]

def main():
    parser = argparse.ArgumentParser(description="RAG Query Interface")
    #parser.add_argument("query", type=str, help="The question to ask." )
    parser.add_argument("--query", type=str, help="The question to ask.", default="What is Python?")
    parser.add_argument("--model", type=str, choices=["ollama", "openrouter"], default="ollama", help="LLM provider to use.")
    args = parser.parse_args()
    
    print(f"Query: {args.query}")
    print(f"Using Model: {args.model}")

    # 1. Vector Search
    print("Retrieving context...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection = get_db_collection()
    
    context_docs = vector_search(args.query, collection, embeddings_model)
    context_text = "\n\n".join(context_docs)
    
    if not context_text:
        print("No relevant context found.")
        return

    # 2. Generate Answer
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    if args.model == "ollama":
        # Ensure 'llama3.1' or your preferred model is pulled
        llm = ChatOllama(model="llama3.1:8b") 
        # llm =ask_ollama()
    elif args.model == "openrouter":
        if not OPENROUTER_API_KEY:
            print("Error: OPENROUTER_API_KEY not found in environment variables.")
            return
        llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="openai/gpt-3.5-turbo" # Or any OpenRouter model
        )
    
    chain = prompt | llm | StrOutputParser()
    
    print("\n--- Answer ---")
    try:
        for chunk in chain.stream({"context": context_text, "question": args.query}):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nError generating answer: {e}")

if __name__ == "__main__":
    main()
