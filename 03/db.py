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

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_CLIENT_KEY,
            model_name=EMBEDDING_MODEL
        )

        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self.embedding_function
        )


    def add_documents(self, documents, metadatas, ids):
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )


    def search(self, query: str, n_results: int = 3):
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )


    def count(self):
        return self.collection.count()


    def remove_document_by_source(self, filename: str):
        self.collection.delete(
            where={"source": filename}
        )


    def get_tracked_files(self):
        """
        Ricostruisce lo stato attuale dei file dal DB
        usando metadata 'source' + hash
        """

        data = self.collection.get(include=["metadatas"])

        tracked = {}

        for meta in data["metadatas"]:
            source = meta["source"]
            tracked[source] = meta

        return tracked