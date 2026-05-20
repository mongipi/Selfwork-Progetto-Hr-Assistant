import os
from dotenv import load_dotenv

load_dotenv(".env")

DOCUMENTS_DIR = "resumes"

OPENAI_CLIENT_KEY = os.environ["OPENAI_CLIENT_KEY"]
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "CVs"
EMBEDDING_MODEL = "text-embedding-3-small"

OLLAMA_MODEL = "llama3.2"