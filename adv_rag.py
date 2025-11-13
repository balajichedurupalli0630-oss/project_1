# -----------------------------
# Groq RAG Chatbot with FAISS + VectorStoreRetrieverMemory
# -----------------------------

import os
import time
import streamlit as st
from pathlib import Path
from typing import List, Any
from dotenv import load_dotenv

# LangChain & Community Imports (v1.0.2+)
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import VectorStoreRetrieverMemory

from langchain_classic.retrievers import MergerRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# 📂 Load Documents
# -----------------------------
def load_all_documents(data: str) -> List[Any]:
    data_path = Path(data).resolve()
    documents = []

    loaders = [
        ("*.pdf", PyPDFLoader),
        ("*.txt", TextLoader),
        ("*.csv", CSVLoader),
        ("*.xlsx", UnstructuredExcelLoader),
        ("*.docx", Docx2txtLoader),
        ("*.json", JSONLoader),
    ]

    for pattern, LoaderClass in loaders:
        for file in data_path.rglob(pattern):
            try:
                loader = LoaderClass(str(file))
                loaded = loader.load()
                documents.extend(loaded)
                print(f"Loaded: {file.name}")
            except Exception as e:
                print(f"[ERROR] Could not load {file}: {e}")

    print(f" Total loaded documents: {len(documents)}")
    return documents


# -----------------------------
#  Streamlit Setup
# -----------------------------
st.set_page_config(page_title="Groq RAG Chatbot", layout="wide")
st.title(" RAG Chatbot with Groq + FAISS Memory (Persistent)")

# -----------------------------
# Initialize Session State
# -----------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = True

    #  Shared embedding model
    st.session_state.embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

    #  Document FAISS (persistent)
    if Path("doc_index").exists():
        st.session_state.doc_store = FAISS.load_local(
            "doc_index",
            st.session_state.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(" Loaded existing document FAISS index.")
    else:
        with st.spinner(" Loading and indexing documents..."):
            docs = load_all_documents("./data")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            st.session_state.doc_store = FAISS.from_documents(chunks, st.session_state.embeddings)
            st.session_state.doc_store.save_local("doc_index")
            print("Document index created and saved.")

    #  Memory FAISS (persistent)
    if Path("memory_index").exists():
        st.session_state.memory_store = FAISS.load_local(
            "memory_index",
            st.session_state.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(" Loaded existing memory FAISS index.")
    else:
        st.session_state.memory_store = FAISS.from_texts([""], st.session_state.embeddings)
        st.session_state.memory_store.save_local("memory_index")

    #  Vector memory retriever (tuned k)
    st.session_state.memory = VectorStoreRetrieverMemory(
        retriever=st.session_state.memory_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 3}
        )
    )

    #  LLM
    st.session_state.llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b"
    )

    #  Prompt
    prompt_template = """
Use the following retrieved context and memory to answer the user's question.
Be clear, natural, and complete.

Context:
{context}

Question: {input}

Answer:
"""
    st.session_state.prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    #  Create RAG chains
    doc_retriever = st.session_state.doc_store.as_retriever(search_kwargs={"k": 4})
    memory_retriever = st.session_state.memory.retriever

    combined_retriever = MergerRetriever(retrievers=[doc_retriever, memory_retriever])

    document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.prompt)
    st.session_state.retrieval_chain = create_retrieval_chain(combined_retriever, document_chain)

    st.success(" Chatbot initialized successfully!")

# -----------------------------
#  Helper Function: Filter Messages
# -----------------------------
def is_relevant_message(text: str) -> bool:
    irrelevant_phrases = ["hi", "hello", "thanks", "bye", "okay"]
    return len(text.strip()) > 5 and not any(p in text.lower() for p in irrelevant_phrases)


query = st.text_input("Ask your question:")

if query:
    start = time.time()

    response = st.session_state.retrieval_chain.invoke({"input": query})
    elapsed = time.time() - start
    answer = response.get("answer", "No answer found.")

    st.markdown(f"**Assistant:** {answer}")
    st.caption(f" Response time: {elapsed:.2f} sec")

    # Save relevant interactions to memory
    if is_relevant_message(query):
        st.session_state.memory.save_context({"input": query}, {"output": answer})
       
        st.session_state.memory_store.save_local("memory_index")


    with st.expander(" Retrieved Memory Context"):
        mem_vars = st.session_state.memory.load_memory_variables({"input": ""})
        st.write(mem_vars)

   
    with st.expander(" Retrieved Documents"):
        for doc in response["context"]:
            st.write(doc.page_content[:400] + "...")
            st.write("---")
