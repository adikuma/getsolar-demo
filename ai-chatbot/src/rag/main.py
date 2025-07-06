from typing import List, Dict, Any
import os
from langchain_core.tools import tool
from .document_loader import load_documents_from_json
from .vector_store import create_vector_store, load_vector_store
from .retrievers import create_hybrid_retriever
from .config import DATA_PATH, PERSIST_DIR
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# global vars
_documents = None
_vector_store = None
_retriever = None

def reset_rag_system():
    global _documents, _vector_store, _retriever
    _documents = _vector_store = _retriever = None
    # print("rag system reset, will re-initialize on next call")

def initialize_rag_system():
    global _documents, _vector_store, _retriever
    if _documents is None or _vector_store is None or _retriever is None:
        # print("initializing rag system...")
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            # print("found existing faiss store, loading...")
            _documents = load_documents_from_json(DATA_PATH)
            _vector_store = load_vector_store()
            _retriever = create_hybrid_retriever(_vector_store, _documents)
            # print("loaded existing rag system successfully")
        else:
            # print("no existing store found, creating new one...")
            _documents = load_documents_from_json(DATA_PATH)
            _vector_store = create_vector_store(_documents)
            _retriever = create_hybrid_retriever(_vector_store, _documents)
            # print("created new rag system successfully")
    else:
        # print("rag system already initialized, using existing components")
        pass
    return _documents, _vector_store, _retriever

def get_context(query: str) -> List[Dict[str, Any]]:
    # print(f"[retrieve_context] called with query snippet: {query[:80]!r}")
    try:
        _, _, retriever = initialize_rag_system()
        results = retriever.get_relevant_documents(query)
        formatted = []
        for i, doc in enumerate(results):
            formatted.append({
                "rank": i+1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("sourceURL","unknown"),
                "title": doc.metadata.get("title","untitled")
            })
        # print(f"[retrieve_context] returning {len(formatted)} docs")
        return formatted
    except Exception as e:
        print(f"[retrieve_context] error: {e}")
        raise

google_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """
you are a precise information assistant. use the context below to answer the 
question without inventing or omitting any details. preserve facts exactly.
list all the details and sources you use to answer the question.
""".strip()

@tool
def rag(query: str) -> str:
    """
    Retrieve relevant information about GetSolar's services, solar technology, and company details.
    
    This tool searches GetSolar's knowledge base to provide accurate, context-grounded answers about:
    • GetSolar company information (plans, pricing, policies, services)
    • Solar panel technology and benefits
    • Installation processes and requirements
    • Singapore-specific solar regulations and incentives
    • Energy savings calculations and examples
    
    IMPORTANT: For ANY GetSolar company-related questions (pricing, plans, policies, contact info, 
    services offered), this tool MUST be used to ensure accurate, up-to-date information from 
    the official knowledge base rather than general knowledge.
    
    Args:
        query (str): User's question about solar energy, GetSolar services, or company information
        
    Returns:
        str: Detailed, context-grounded answer with preserved facts and source references
    """
    # print(f"[rag_answer] called with query snippet: {query[:80]!r}")
    try:
        _, _, retriever = initialize_rag_system()
        # docs = retriever.get_relevant_documents(query)
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(
            f"[{i+1}] {d.metadata.get('title','')}\n{d.page_content}"
            for i, d in enumerate(docs[:5])
        )
        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        # print(f"[rag_answer] sending prompt snippet: {prompt[:200]!r}")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        resp = google_llm.invoke(messages)
        # print("[rag_answer] received response")
        return resp.content
    except Exception as e:
        print(f"[rag_answer] error: {e}")
        raise

def test_retrieval(query: str, max_results: int = 3):
    print(f"query: {query}")
    print("-" * 50)
    results = rag({"query": query})
    print(f"found {len(results)} relevant documents\n")
    for r in results[:max_results]:
        print(f"document {r['rank']}:")
        print(f" source: {r['source']}")
        print(f" title:  {r['title']}")
        print(f" content:{r['content'][:200]}...")
        print()
        
def test_rag_answer(query: str):
    print(f"query: {query}")
    print("-" * 50)
    try:
        answer = rag.invoke({"query": query})
        print("answer:")
        print(answer)
    except Exception as e:
        print("error during RAG answer:", e)
    print()


if __name__ == "__main__":
    test_queries = [
        "getsolar rent to own program",
        "solar panel installation cost singapore",
        "hdb flat solar panels eligibility"
    ]
    for query in test_queries:
        test_rag_answer(query)
        print("=" * 60, "\n")