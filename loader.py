import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the path to the directory containing the documents
DATA_PATH = "documents/"

def load_and_chunk_documents():
    """
    Loads documents from the specified directory and splits them into chunks.
    
    Returns:
        list: A list of document chunks.
    """
    print("Loading documents...")
    # Initialize the DirectoryLoader to load files from the 'documents' folder
    # It automatically selects the correct loader for each file type.
    loader = DirectoryLoader(DATA_PATH, glob="**/*", show_progress=True)
    
    documents = loader.load()
    
    if not documents:
        print("No documents found in the specified directory.")
        return []

    print(f"Loaded {len(documents)} document(s).")
    
    # Initialize the text splitter
    # chunk_size is the max number of characters in a chunk
    # chunk_overlap is the number of characters to overlap between chunks
    # to maintain context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    print("Chunking documents...")
    chunked_docs = text_splitter.split_documents(documents)
    
    print(f"Successfully split documents into {len(chunked_docs)} chunks.")
    
    return chunked_docs

# --- Main execution ---
if __name__ == "__main__":
    # Run the function
    all_chunks = load_and_chunk_documents()
    
    # You can inspect the first chunk to see what it looks like
    if all_chunks:
        print("\n--- Example of the first chunk ---")
        print(all_chunks[0].page_content)
        print("\n--- Metadata of the first chunk ---")
        print(all_chunks[0].metadata)