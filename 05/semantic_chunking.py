import re
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from config import OPENAI_CLIENT_KEY


class SemanticChunking:

    # -------------------------
    # COMBINE CONTEXT
    # -------------------------
    @staticmethod
    def combine_sentences(sentences, buffer_size=1):

        for i in range(len(sentences)):

            combined = []

            for j in range(i - buffer_size, i + buffer_size + 1):

                if 0 <= j < len(sentences):
                    combined.append(sentences[j]["sentence"])

            sentences[i]["combined_sentences"] = " ".join(combined)

        return sentences

    # -------------------------
    # COSINE SIMILARITY
    # -------------------------
    @staticmethod
    def cosine_similarity(v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)

        return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

    # -------------------------
    # DISTANCES
    # -------------------------
    @staticmethod
    def calculate_cosine_distances(sentences):

        distances = []

        for i in range(len(sentences) - 1):

            sim = SemanticChunking.cosine_similarity(
                sentences[i]["embedding"],
                sentences[i + 1]["embedding"]
            )

            distances.append(1 - sim)

        return distances, sentences

    # -------------------------
    # MAIN CHUNKING
    # -------------------------
    @staticmethod
    def chunk_it(text):

        # 1. split sentences
        raw_sentences = re.split(r"(?<=[.?!])\s+", text)

        sentences = [
            {"sentence": s, "index": i}
            for i, s in enumerate(raw_sentences)
            if s.strip()
        ]

        # 2. add context
        sentences = SemanticChunking.combine_sentences(sentences)

        # 3. embeddings
        embedder = OpenAIEmbeddingFunction(
            api_key=OPENAI_CLIENT_KEY,
            model_name="text-embedding-3-small"
        )

        texts = [s["combined_sentences"] for s in sentences]
        embeddings = embedder(texts)

        for i, s in enumerate(sentences):
            s["embedding"] = embeddings[i]

        # 4. distances
        distances, sentences = SemanticChunking.calculate_cosine_distances(sentences)

        # 5. threshold
        threshold = np.percentile(distances, 95)

        breakpoints = [
            i for i, d in enumerate(distances)
            if d > threshold
        ]

        # 6. build chunks
        chunks = []
        start = 0

        for bp in breakpoints:

            group = sentences[start:bp + 1]
            chunks.append(" ".join([g["sentence"] for g in group]))
            start = bp + 1

        # last chunk
        if start < len(sentences):
            chunks.append(" ".join([s["sentence"] for s in sentences[start:]]))

        return chunks