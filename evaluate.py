import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# This script is for command-line testing and evaluation.
# It allows you to see the retrieved context for each question.

def initialize_components(pinecone_index_name, embedding_model_name, llm_model_name):
    """Initializes and caches all major components."""
    llm = ChatGroq(model_name=llm_model_name, temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
    vector_store = PineconeVectorStore.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    return llm, retriever

def get_rag_chain(_retriever, _llm):
    """Creates a conversational RAG chain."""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(_llm, _retriever, contextualize_q_prompt)

    qa_system_prompt = (
        "You are an expert assistant designed to provide accurate, detailed, and helpful responses based solely on the retrieved context below. "
        "Use the provided information to construct a clear, concise, and well-organized answer to the user's query. "
        "Respond in a professional and informative tone. Where appropriate, use bullet points, lists, or headings to enhance readability. "
        "Do not refer to or mention the existence of context documents. "
        "If the answer cannot be found within the context, respond with: 'The information required to answer this question is not available in the provided documents.'"
        "\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def main():
    """Main function to run the command-line evaluation tool."""
    load_dotenv()
    
    print("--- RAG System Evaluation Tool ---")
    print("Initializing components...")
    
    try:
        PINECONE_INDEX_NAME = "digire"
        EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        LLM_MODEL_NAME = "llama3-8b-8192"
        
        llm, retriever = initialize_components(PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME)
        rag_chain = get_rag_chain(retriever, llm)
        print("✅ System Initialized. You can now ask questions.")
    except Exception as e:
        print(f"❌ Failed to initialize. Please check your API keys. Details: {e}")
        return

    chat_history = []
    
    while True:
        user_question = input("\nAsk a question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break

        # Invoke the chain with history
        response = rag_chain.invoke({"input": user_question, "chat_history": chat_history})
        answer = response["answer"]
        
        # --- Terminal Logging Output ---
        print("\n" + "="*80)
        print(f"QUESTION: {user_question}")
        print("\n--- RETRIEVED CONTEXT ---")
        for i, chunk in enumerate(response.get('context', [])):
            source_file = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            print(f"\n[CHUNK {i+1} from {source_file}]:")
            print(chunk.page_content)
        print("\n" + "-"*80)
        print(f"LLM ANSWER: {answer}")
        print("="*80)
        
        # Update chat history for the next turn
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))

    print("\n--- Evaluation session ended. ---")

if __name__ == "__main__":
    main()