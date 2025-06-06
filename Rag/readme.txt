# 📚 Document Q&A App with Gemini + LangChain + FAISS

This is a simple Streamlit web application that allows users to upload a `.txt` file and ask questions based on its content. The app uses:

- **Google Gemini (via LangChain)** for language generation
- **FAISS** for semantic search and vector storage
- **HuggingFace Embeddings** for converting text to vectors
- **LangChain** to create a Retrieval-Augmented Generation (RAG) pipeline

---

## 🚀 Features

- Upload any `.txt` document
- Automatically splits and embeds the text
- Asks natural language questions based on the file
- Uses RetrievalQA with Gemini LLM to generate contextual answers

---

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
