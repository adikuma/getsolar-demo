import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .config import OPENAI_API_KEY, EMBEDDING_MODEL, PERSIST_DIR


def create_vector_store(documents: List[Document]) -> Chroma:
    # initialize embedding model
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, 
        openai_api_key=OPENAI_API_KEY
    )
    
    # create persist directory
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # create chroma vector store
    # persist_directory expects a string path to avoid mixing Path and str internally
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR)
    )
    
    print(f"added {len(documents)} documents to chroma vector store")
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, 
        openai_api_key=OPENAI_API_KEY
    )
    
    # ensure persist_directory is a string to prevent Path + str errors
    vector_store = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings
    )
    
    print("loaded existing chroma vector store")
    return vector_store