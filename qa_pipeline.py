import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import json
import datetime
from collections import defaultdict



#load_dotenv(dotenv_path=")"

def load_documents(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory="vectorstore")
    return db  # db.persist() no longer needed

def ask_question(vectorstore, question):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa_chain.invoke(question)  # Updated for LangChain 0.2+

if __name__ == "__main__":
    docs = load_documents("documents")
    vector_db = create_vector_store(docs)

    query = input("Enter your question: ")
    result = ask_question(vector_db, query)
    print("\n💡 Answer:\n", result["result"])

    source_pages = defaultdict(set)
    for doc in result["source_documents"]:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        if source is not None and page is not None:
            source_pages[source].add(page)
        
    print("\n Sources With Pages:")
    for source, pages in source_pages.items():
        page_list = sorted(pages)
        print(f"- {source} -> Pages: {page_list}")

    with open("memory.jsonl", "a") as log:
        log.write(json.dumps({
            "timestamp": str(datetime.datetime.now()),
            "question": query,
            "answer": result["result"],
        }) + "\n")
