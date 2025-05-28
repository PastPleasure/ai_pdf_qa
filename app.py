import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from collections import defaultdict
import json, datetime
from qa_pipeline import load_documents, create_vector_store, ask_question
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI PDF Synthesizer", layout="wide")

st.title("AI PDF Knowledge Synthesizer")
st.markdown("Upload one or more PDF files and ask a question based on their content.")

uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s).")
    os.makedirs("documents", exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())

    st.info("Files saved to `documents/` directory")

question = st.text_input("Ask a question")
if question:
    with st.spinner("Thinking..."):
        docs = load_documents("documents")
        vector_db = create_vector_store(docs)
        result = ask_question(vector_db, question)

        st.markdown("### 💡 Answer:")
        st.write(result["result"])
        
        st.markdown("### 📄 Source Documents:")
        source_pages = defaultdict(set)
        for doc in result["source_documents"]:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page")
            if source and page is not None:
                source_pages[source].add(page)
        for source, pages in source_pages.items():
            st.write(f"- `{source}` → Pages: {sorted(pages)}")

