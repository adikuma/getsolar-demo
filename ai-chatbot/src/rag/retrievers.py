from typing import List
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .config import FAISS_K, BM25_K, ENSEMBLE_WEIGHTS

def create_hybrid_retriever(vector_store: FAISS, documents: List[Document]) -> EnsembleRetriever:
    # create faiss retriever for semantic search
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FAISS_K}
    )
    
    # create bm25 retriever for keyword search
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = BM25_K
    
    # create ensemble retriever combining both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], 
        weights=ENSEMBLE_WEIGHTS
    )
    
    print("created hybrid retriever with faiss and bm25")
    return ensemble_retriever


def get_faiss_retriever(vector_store: FAISS) -> object:
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FAISS_K}
    )


def get_bm25_retriever(documents: List[Document]) -> BM25Retriever:
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = BM25_K
    return bm25_retriever