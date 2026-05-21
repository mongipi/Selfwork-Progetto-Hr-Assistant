import re
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from config import OPENAI_CLIENT_KEY


class SemanticChunking:

    def __init__(self, buffer_size: int = 1, breakpoint_percentile: int = 95):
        self.buffer_size = buffer_size
        self.breakpoint_percentile = breakpoint_percentile
        self.embedder = OpenAIEmbeddingFunction(
            api_key=OPENAI_CLIENT_KEY,
            model_name="text-embedding-3-small"
        )

    def _combine_sentences(self, sentences):

        for i in range(len(sentences)):

            combined = []

            for j in range(i - self.buffer_size, i + self.buffer_size + 1):

                if 0 <= j < len(sentences):
                    combined.append(sentences[j]["sentence"])

            sentences[i]["combined_sentences"] = " ".join(combined)

        return sentences

    def _cosine_similarity(self, v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)

        return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

    def _calculate_cosine_distances(self, sentences):

        distances = []

        for i in range(len(sentences) - 1):

            sim = self._cosine_similarity(
                sentences[i]["embedding"],
                sentences[i + 1]["embedding"]
            )

            distances.append(1 - sim)

        return distances, sentences

    def chunk_text(self, text):

        # 1. split sentences
        raw_sentences = re.split(r"(?<=[.?!])\s+", text)

        sentences = [
            {"sentence": s, "index": i}
            for i, s in enumerate(raw_sentences)
            if s.strip()
        ]

        # 2. add context
        sentences = self._combine_sentences(sentences)

        # 3. embeddings
        texts = [s["combined_sentences"] for s in sentences]
        embeddings = self.embedder(texts)

        for i, s in enumerate(sentences):
            s["embedding"] = embeddings[i]

        # 4. distances
        distances, sentences = self._calculate_cosine_distances(sentences)

        # 5. threshold
        threshold = np.percentile(distances, self.breakpoint_percentile)

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

        if start < len(sentences):
            chunks.append(" ".join([s["sentence"] for s in sentences[start:]]))

        return chunks