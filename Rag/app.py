import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os

st.title("📄 Document Q&A with Gemini")

os.environ["GOOGLE_API_KEY"] = "AIzaSyDGOdBnd43a8YKza0V3jzKBnCdedoUjVH0"

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

@st.cache_resource
def load_and_process_document(text):
    """Splits text and creates vector store."""
    try:
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=0,
        )
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(
            [Document(page_content=chunk) for chunk in chunks], embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error during vectorstore creation: {e}")
        return None

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        vectorstore = load_and_process_document(file_content)

        if vectorstore:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

            query = st.text_input("Ask a question based on the uploaded document:")

            if query:
                with st.spinner("Getting answer..."):
                    try:
                        answer = qa_chain.run(query)
                        st.write("### Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error getting answer: {e}")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
else:
    st.info("Please upload a `.txt` file to begin.")
