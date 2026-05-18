import chromadb
from chromadb.utils import embedding_functions

from config import (
    OPENAI_CLIENT_KEY,
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL
)


class CVDatabase:
    def __init__(self):
        # Embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_CLIENT_KEY,
            model_name=EMBEDDING_MODEL
        )

        # Persistent DB
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self.embedding_function
        )

    # -----------------------------------------------------
    # ADD / INDEX
    # -----------------------------------------------------
    def add_documents(self, documents, metadatas, ids):
        if self.collection.count() == 0:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("CV indicizzati correttamente.")
        else:
            print("Database già popolato.")

    # -----------------------------------------------------
    # QUERY
    # -----------------------------------------------------
    def search(self, query: str, n_results: int = 3):
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    # -----------------------------------------------------
    # UTILE (opzionale)
    # -----------------------------------------------------
    def count(self):
        return self.collection.count()