AI PDF Knowledge Synthesizer
Upload one or multiple PDF documents, ask any question, and get an AI-generated answer with source references including filenames and page numbers. Built with LangChain, OpenAI, Streamlit, and ChromaDB.

FEATURES
Upload one or more PDF files

Parse and chunk text using LangChain

Store embeddings in ChromaDB

Ask natural language questions

Retrieve relevant info via semantic search

Answer generated by GPT-3.5 or GPT-4

Source output includes filename and page numbers

Q&A history logged to memory.jsonl

Fully containerized with Docker

TECH STACK
Streamlit (frontend)

LangChain (retriever + embeddings + QA chain)

Chroma (local vector database)

OpenAI API (answer generation)

PyPDFLoader (PDF parsing)

dotenv (for key management)

Docker (for deployment)

FOLDER STRUCTURE
ai_pdf_qa/
├── app.py ← Streamlit web UI
├── qa_pipeline.py ← Core logic (load, embed, ask)
├── documents/ ← Uploaded PDFs
├── vectorstore/ ← ChromaDB vector store
├── memory.jsonl ← Appended Q&A logs
├── Dockerfile ← Docker config
├── .dockerignore ← Ignore unnecessary files
├── requirements.txt ← Dependencies
├── README.md

HOW TO RUN LOCALLY
Clone the repo

git clone https://github.com/yourusername/ai_pdf_qa.git
cd ai_pdf_qa

Install dependencies

pip install -r requirements.txt

Create a file named .env and add your OpenAI API key

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

Start the app

streamlit run app.py
