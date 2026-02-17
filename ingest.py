import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "rag_db"
COLLECTION_NAME = "Rag_doc"
DATA_DIR = "data"

def load_documents():
    documents = []
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        return []
    
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def ingest_data():
    # 1. Load Data
    docs = load_documents()
    if not docs:
        print("No documents to ingest.")
        return

    # 2. Split Data
    chunks = split_documents(docs)

    # 3. Generate Embeddings & Store in MongoDB
    print("Initializing Embeddings (sentence-transformers)...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Clear existing data (optional but good for testing)
    collection.delete_many({})
    print("Cleared existing data.")

    print("Generating embeddings and inserting into MongoDB...")
    docs_to_insert = []
    for chunk in chunks:
        # Generate embedding
        vector = embeddings_model.embed_query(chunk.page_content)
        
        # Prepare document
        doc = {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": vector
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        collection.insert_many(docs_to_insert)
        print(f"Successfully inserted {len(docs_to_insert)} documents with embeddings.")
    else:
        print("No documents to insert.")

if __name__ == "__main__":
    ingest_data()
