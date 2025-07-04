import json
from typing import List
from langchain_core.documents import Document

def load_documents_from_json(file_path: str) -> List[Document]:
    documents = []
    
    # load json data
    with open(file_path, "r") as f:
        raw = json.load(f)
    
    # process each page in the data
    for page in raw["data"]:
        content = page.get("markdown", page.get("content", ""))
        metadata = page.get("metadata", {})
        
        # create document with metadata
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"created {len(documents)} documents with metadata")
    return documents