import os
import uuid
from config import DOCUMENTS_DIR
from db import CVDatabase


def index_resumes(db: CVDatabase):

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(DOCUMENTS_DIR):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(DOCUMENTS_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read().replace("\n", " ")
            chunks = text.split("### ")

            for chunk in chunks:
                chunk = chunk.strip()

                if chunk:
                    documents.append(chunk)
                    metadatas.append({"source": filename})
                    ids.append(str(uuid.uuid4()))

    db.add_documents(documents, metadatas, ids)