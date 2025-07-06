import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

HERE = Path(__file__).parent
ROOT = HERE.parent.parent  

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_PATH = ROOT / "data" / "scrape_result.json"
PERSIST_DIR = HERE / "faiss_store"

EMBEDDING_MODEL = "text-embedding-ada-002"

FAISS_K = 5
BM25_K = 5
ENSEMBLE_WEIGHTS = [0.8, 0.2]  # faiss weight, bm25 weight