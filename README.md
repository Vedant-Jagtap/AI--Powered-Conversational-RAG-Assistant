# 📄 Multi-PDF Conversational RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents and interact with them using natural language queries.

Built using **Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Google Gemini**.

---

## 🚀 Live Demo

**Deployment Link:** https://pdf-conversational-rag-assistant-v2.streamlit.app/

## 📌 Features

### 📚 Multi-PDF Support

* Upload multiple PDF documents simultaneously.
* Create a unified knowledge base across all uploaded files.

### 🔍 Single PDF Search Mode

* Query a specific PDF document.
* Useful when working with large document collections.

### 🌐 Multi-PDF Search Mode

* Search across all uploaded documents at once.
* Retrieve information regardless of which document contains it.

### 💬 Conversational Memory

* Supports follow-up questions.
* Maintains conversation context for more natural interactions.

### 📄 Source Attribution

* Displays which PDF(s) were used to generate the answer.
* Improves transparency and trustworthiness.

### ⚡ Semantic Search

* Uses vector embeddings and FAISS similarity search.
* Retrieves contextually relevant information rather than keyword matching.

### 🤖 Gemini-Powered Responses

* Generates accurate answers using Google Gemini.
* Responses are grounded in retrieved document content.

### 🔄 Dynamic PDF Processing

* PDFs are processed into chunks automatically.
* Vector database is generated on demand.

### 🧠 Cross-Document Understanding

* Compare documents.
* Summarize multiple PDFs.
* Identify relationships between uploaded files.

---

## 🏗️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### LLM

* Google Gemini 2.5 Flash

### Frameworks

* LangChain

### Vector Database

* FAISS

### Embedding Model

* sentence-transformers/all-MiniLM-L6-v2

### PDF Processing

* PyPDF

---

## ⚙️ How It Works

1. Upload one or more PDF files.
2. Text is extracted from the PDFs.
3. Documents are split into chunks.
4. Embeddings are generated using HuggingFace models.
5. Chunks are stored in a FAISS vector database.
6. User asks a question.
7. Relevant chunks are retrieved.
8. Retrieved context is sent to Gemini.
9. Gemini generates an answer grounded in the document content.

---

## 📂 Project Structure

```text
AI-Powered-Conversational-RAG-Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── faiss_index/
├── vector_store/
│
└── assets/
```

---

## 🖥️ Installation

Clone the repository:

```bash
git clone https://github.com/Vedant-Jagtap/AI--Powered-Conversational-RAG-Assistant.git
```

Move into the project folder:

```bash
cd AI--Powered-Conversational-RAG-Assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Example Use Cases

### Education

* Study notes assistant
* Research paper analysis
* Multi-document summarization

### Business

* Policy document search
* Report comparison
* Knowledge management

### Personal Productivity

* Resume analysis
* Certificate tracking
* Travel planning document search

---

## 🔮 Future Enhancements

* Chat export functionality
* PDF citation highlighting
* Document metadata filtering
* OCR support for scanned PDFs
* Multi-format support (DOCX, TXT, PPTX)
* User authentication
* Cloud vector database integration

---

## 👨‍💻 Author

**Vedant Jagtap**

GitHub:
https://github.com/Vedant-Jagtap

---

## ⭐ If you found this project useful

Consider giving the repository a star.
