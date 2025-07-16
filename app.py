import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Load Environment Variables (do this once) ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Corporate Intelligence Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- CSS Implementing YOUR Exact Visual Style, But Correctly ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global variables from your provided CSS */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
        --border-radius: 16px;
        --text-color: #FFFFFF;
    }

    body {
        font-family: 'Inter', sans-serif;
    }

    /* Main app background with YOUR animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: var(--text-color);
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* CRITICAL FIX: Make the main container transparent to prevent empty space */
    .main .block-container {
        background: transparent;
        padding-top: 2rem;
    }
    
    /* Centered Welcome Message Styling */
    .welcome-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70vh;
    }
    .welcome-container h1 {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: var(--text-color);
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    /* Apply YOUR glass effect to the sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
     [data-testid="stSidebar"] * {
        color: var(--text-color);
    }
    
    /* Apply YOUR gradient to the button */
    .stButton > button {
        background: var(--primary-gradient);
        color: var(--text-color);
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px 0 rgba(31, 38, 135, 0.5);
    }

    /* Apply YOUR glass effect to chat bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: var(--border-radius);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-light);
        animation: fadeIn 0.5s ease-in-out;
    }

    /* Apply a glass effect to the chat input area */
    [data-testid="stChatInputContainer"] {
        background: transparent !important;
        border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.25) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: var(--border-radius) !important;
    }
    [data-testid="stChatInputSubmitButton"] svg {
        fill: var(--text-color);
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.1) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
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
    # --- Sidebar ---
    with st.sidebar:
        st.header("AI-Powered Corporate Intelligence Assistant")
        st.markdown("This chatbot provides answers based on a curated set of internal documents.")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- CENTRALIZED INITIALIZATION (runs only once) ---
    if "rag_chain" not in st.session_state:
        with st.spinner("Initializing system..."):
            try:
                PINECONE_INDEX_NAME = "digire"
                EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
                LLM_MODEL_NAME = "llama3-8b-8192"
                llm, retriever = initialize_components(PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME)
                st.session_state.rag_chain = get_rag_chain(retriever, llm)
            except Exception as e:
                st.error(f"Failed to initialize. Please check API keys. Details: {e}")
                st.stop()
    
    # --- Chat History & UI Display Logic ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Conditional display: Show welcome message or chat history
    if not st.session_state.messages:
        st.markdown('<div class="welcome-container"><h1>Ask a question about your documents</h1></div>', unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.info(source)

    # --- Chat Input & Conversational Logic ---
    if user_question := st.chat_input("What would you like to know?"):
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Prepare history and get assistant response
        chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages[:-1]]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_question, "chat_history": chat_history})
                    answer = response["answer"]
                    sources = [f"{os.path.basename(doc.metadata.get('source', 'N/A'))} (Page: {doc.metadata.get('page', 'N/A')})" for doc in response.get("context", [])]
                    
                    st.markdown(answer)
                    with st.expander("View Sources"):
                        if sources:
                            for source in sources:
                                st.info(source)
                        else:
                            st.write("No specific sources were retrieved for this answer.")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()