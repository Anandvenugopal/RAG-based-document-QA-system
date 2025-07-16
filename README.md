# 📄 RAG-Based Document QA System (with Pinecone + LLaMA 3)

This project is an intelligent, RAG-based (Retrieval-Augmented Generation) chatbot that allows you to ask questions across multiple document types like PDF, PPTX, DOCX, etc. It uses **LLaMA 3 via Groq API** for natural language understanding and **Pinecone** for efficient vector search.

## 🔧 Features

- ✅ Multi-format document support (PDF, PPTX, DOCX, etc.)
- ✅ Automatic chunking and embedding
- ✅ Uses `sentence-transformers/all-MiniLM-L6-v2` for fast and efficient embeddings
- ✅ Powered by LLaMA 3 (via Groq)
- ✅ Real-time conversational interface using Streamlit
- ✅ Fully integrated with Pinecone Vector Database

---

## 🗂 Project Structure

```
├── app.py              # Streamlit UI for chat-based Q&A
├── loader.py           # Document loader and Pinecone vector ingestion
├── requirements.txt    # Python dependencies
├── documents/          # Directory to store all input documents
├── .env                # Stores API keys and environment configuration
├── EVALUATION.md
├── evaluate.py
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Anandvenugopal/RAG-based-document-QA-system.git
cd RAG-based-document-QA-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory and add the following:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

```

🔑 Get your API keys from:

- [Groq Console](https://console.groq.com/)
- [Pinecone Console](https://www.pinecone.io/)

---

### 4. Add Your Documents

Place your source files (`.pdf`, `.docx`, `.pptx`, etc.) in the `documents/` folder.

---

### 5. Ingest and Index Documents

This will chunk your documents, create embeddings, and upload them to Pinecone:

```bash
python loader.py
```

---

### 6. Launch the Chat Interface

Run the Streamlit app:

```bash
streamlit run app.py
```

Now open your browser at [http://localhost:8501](http://localhost:8501) to interact with the system.

---

## 🤖 Model & Tools Used

- **LLM:** LLaMA 3 (`llama3-8b-8192`) via [Groq API](https://console.groq.com/)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB:** [Pinecone](https://www.pinecone.io/)
- **Frontend:** [Streamlit](https://streamlit.io/)

---

## 📌 License

This project is for educational and research use. Please check with relevant APIs and services for their usage terms.

---

> Built with ❤️ by [Anand Venugopal](https://github.com/Anandvenugopal)
