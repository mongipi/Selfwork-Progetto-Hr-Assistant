import re
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

#
# Spiegazione Generale della Logica:
# Questo codice implementa un sistema di "semantic chunking", ovvero un modo intelligente di dividere un testo lungo in parti più piccole (chunk) che hanno senso semantico. Invece di dividere il testo in modo meccanico (per esempio ogni 500 caratteri), questo sistema usa l'intelligenza artificiale per capire dove il significato del testo cambia significativamente. Lo fa analizzando ogni frase nel suo contesto e utilizzando gli embedding di OpenAI per misurare quanto due frasi consecutive sono semanticamente diverse. Quando trova un punto dove la differenza semantica è particolarmente alta (sopra il 95° percentile per default), usa quel punto come confine per creare un nuovo chunk. Questo approccio è particolarmente utile per sistemi di elaborazione del linguaggio naturale, come chatbot o sistemi di ricerca, perché permette di mantenere insieme le parti di testo che sono semanticamente correlate.
#


class SemanticChunking:
    def __init__(self, api_key, breakpoint_percentile=95, buffer_size=1):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=Config.MODEL_NAME)
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size

    def _split_into_sentences(self, text):
        # First try to split by common sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        # If we only got one sentence and it's longer than 100 characters,
        # fall back to splitting by other common delimiters
        if len(sentences) == 1 and len(text) > 100:
            # Split by common delimiters while preserving them
            delimiters = r"([.!?\n;:])"
            parts = re.split(delimiters, text.strip())

            # Recombine the parts with their delimiters
            sentences = []
            for i in range(0, len(parts) - 1, 2):
                if parts[i].strip():  # Only add non-empty sentences
                    sentences.append(parts[i].strip() + parts[i + 1])

            # If we still have only one sentence, split by comma as last resort
            if len(sentences) == 1:
                sentences = [s.strip() + "," for s in text.split(",") if s.strip()]
                if sentences:  # Replace the last comma with a period
                    sentences[-1] = sentences[-1][:-1] + "."

        # Ensure we have at least one non-empty sentence
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            sentences = [text + "."]

        return sentences

    def _process_sentences(self, text):
        # Use the improved sentence splitting
        raw_sentences = self._split_into_sentences(text)

        # Create list of dictionaries with indices
        sentences = [{"sentence": s, "index": i} for i, s in enumerate(raw_sentences)]

        # Combine each sentence with its context
        for i, current in enumerate(sentences):
            context_range = range(
                max(0, i - self.buffer_size),
                min(len(sentences), i + self.buffer_size + 1),
            )
            current["combined_sentence"] = " ".join(
                sentences[j]["sentence"] for j in context_range
            )

        return sentences

    def _calculate_distances(self, sentences):
        # Calcola gli embedding per tutte le frasi combinate
        embeddings = self.embeddings.embed_documents(
            [s["combined_sentence"] for s in sentences]
        )

        # Calcola le distanze coseno tra frasi consecutive
        distances = []
        for i in range(len(sentences) - 1):
            distance = 1 - cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distances.append(distance)

        return distances

    def chunk_text(self, text):
        # Process sentences
        sentences = self._process_sentences(text)

        # If we have no valid sentences, return the original text as a single chunk
        if not sentences:
            return [text]

        # Calculate distances
        distances = self._calculate_distances(sentences)

        # Determine split points based on percentile
        threshold = np.percentile(distances, self.breakpoint_percentile)
        split_points = [i for i, d in enumerate(distances) if d > threshold]

        # Create final chunks
        chunks = []
        start = 0
        for point in split_points + [len(sentences) - 1]:
            chunk = " ".join(s["sentence"] for s in sentences[start : point + 1])
            chunks.append(chunk)
            start = point + 1

        return chunks
