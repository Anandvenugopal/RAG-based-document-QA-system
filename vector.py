import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
DATA_PATH = "documents/"
PINECONE_INDEX_NAME = "digire" # Make sure this matches your new index name
# --- Switched to the smaller, faster model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_store():
    """
    Loads docs, splits them, creates 384-dimension embeddings, and stores in Pinecone.
    """
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*", show_progress=True, use_multithreading=True, silent_errors=True)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s). Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Successfully split documents into {len(chunked_docs)} chunks.")

    print(f"Creating embeddings with '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("Embeddings model loaded.")

    print(f"Ingesting {len(chunked_docs)} chunks into Pinecone index '{PINECONE_INDEX_NAME}'...")
    PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    create_vector_store()