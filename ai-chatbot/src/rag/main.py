from typing import List, Dict, Any
from langchain_core.tools import tool
from document_loader import load_documents_from_json
from vector_store import create_vector_store, load_vector_store
from retrievers import create_hybrid_retriever
from config import DATA_PATH, PERSIST_DIR


# global variables for storing initialized components
_documents = None
_vector_store = None
_retriever = None

def reset_rag_system():
    global _documents, _vector_store, _retriever
    _documents = None
    _vector_store = None
    _retriever = None
    print("rag system reset, will re-initialize on next call")

def initialize_rag_system():
    global _documents, _vector_store, _retriever
    
    if _documents is None or _vector_store is None or _retriever is None:
        print("initializing rag system...")
        
        # check if faiss store already exists
        import os
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            print("found existing faiss store, loading...")
            _documents = load_documents_from_json(DATA_PATH)
            _vector_store = load_vector_store()
            _retriever = create_hybrid_retriever(_vector_store, _documents)
            print("loaded existing rag system successfully")
            
        else:
            print("no existing store found, creating new one...")
            _documents = load_documents_from_json(DATA_PATH)
            _vector_store = create_vector_store(_documents)
            _retriever = create_hybrid_retriever(_vector_store, _documents)
            
            print("created new rag system successfully")
    else:
        print("rag system already initialized, using existing components")
    
    return _documents, _vector_store, _retriever


@tool
def retrieve_context(query: str) -> List[Dict[str, Any]]:
    """
    retrieves relevant context documents for a given query using hybrid search
    
    args:
        query (str): user question or search query
        
    returns:
        List[Dict[str, Any]]: list of relevant documents with content and metadata
    """
    _, _, retriever = initialize_rag_system()
    
    results = retriever.get_relevant_documents(query)
    
    formatted_results = []
    for i, doc in enumerate(results):
        formatted_results.append({
            "rank": i + 1,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", "untitled")
        })
    
    return formatted_results


def test_retrieval(query: str, max_results: int = 3):
    print(f"query: {query}")
    print("-" * 50)
    
    results = retrieve_context.invoke({"query": query})
    
    print(f"found {len(results)} relevant documents")
    print()
    
    for result in results[:max_results]:
        print(f"document {result['rank']}:")
        print(f"source: {result['source']}")
        print(f"title: {result['title']}")
        print(f"content: {result['content'][:200]}...")
        print()


if __name__ == "__main__":
    test_queries = [
        "getsolar rent to own program",
        "solar panel installation cost singapore",
        "hdb flat solar panels eligibility"
    ]
    
    # run tests
    for query in test_queries:
        test_retrieval(query)
        print("=" * 60)
        print()