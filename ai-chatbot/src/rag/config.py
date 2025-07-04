import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_PATH = "../../data/scrape_result.json"
PERSIST_DIR = "faiss_store"

EMBEDDING_MODEL = "text-embedding-ada-002"

FAISS_K = 5
BM25_K = 5
ENSEMBLE_WEIGHTS = [0.8, 0.2]  # faiss weight, bm25 weight