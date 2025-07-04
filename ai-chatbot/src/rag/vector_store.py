import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import OPENAI_API_KEY, EMBEDDING_MODEL, PERSIST_DIR


def create_vector_store(documents: List[Document]) -> Chroma:
    # initialize embedding model
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, 
        openai_api_key=OPENAI_API_KEY
    )
    
    # create persist directory
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # create chroma vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    print(f"added {len(documents)} documents to chroma vector store")
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, 
        openai_api_key=OPENAI_API_KEY
    )
    
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    print("loaded existing chroma vector store")
    return vector_store