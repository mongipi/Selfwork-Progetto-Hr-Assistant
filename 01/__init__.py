import os
import uuid

import chainlit as cl
import chromadb
import ollama

from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================

load_dotenv(".env")

DOCUMENTS_DIR = "resumes"

# =========================================================
# CHROMA DB
# =========================================================

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_CLIENT_KEY"],
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="CVs",
    embedding_function=openai_ef
)

# =========================================================
# LETTURA E INDICIZZAZIONE CV
# =========================================================

documents = []
metadatas = []
ids = []

for filename in os.listdir(DOCUMENTS_DIR):

    if filename.endswith(".txt"):

        filepath = os.path.join(DOCUMENTS_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as file:

            text = file.read().replace("\n", " ")

            # Chunking semplice
            chunks = text.split("### ")

            for chunk in chunks:

                chunk = chunk.strip()

                if chunk:

                    documents.append(chunk)

                    metadatas.append({
                        "source": filename
                    })

                    ids.append(str(uuid.uuid4()))

# Evita duplicati ad ogni riavvio
if collection.count() == 0:

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("CV indicizzati correttamente.")

else:
    print("Database già popolato.")

# =========================================================
# FUNZIONI UTILI
# =========================================================

def leggi_prime_righe(file_path, n=20):

    righe = []

    with open(file_path, "r", encoding="utf-8") as file:

        for i, riga in enumerate(file):

            if i < n:
                righe.append(riga.strip())
            else:
                break

    return "\n".join(righe)

# =========================================================
# CHAT START
# =========================================================

@cl.on_chat_start
async def on_chat_start():

    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": """
Sei un assistente HR specializzato nella selezione del personale.

Il tuo compito è:
- individuare il candidato più adatto
- motivare la scelta
- usare solo le informazioni presenti nei CV
- non inventare informazioni
- rispondere in modo professionale e sintetico
"""
            }
        ]
    )

    await cl.Message(
        content="Ciao! Sono il tuo HR Assistant 🤖"
    ).send()

# =========================================================
# GESTIONE MESSAGGI
# =========================================================

@cl.on_message
async def handle_message(message: cl.Message):

    user_question = message.content

    # =====================================================
    # RICERCA SEMANTICA CV
    # =====================================================

    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )

    contexts = []

    for i in range(len(results["documents"][0])):

        document = results["documents"][0][i]

        metadata = results["metadatas"][0][i]

        filename = metadata["source"]

        filepath = os.path.join(DOCUMENTS_DIR, filename)

        candidate_info = leggi_prime_righe(filepath, 20)

        contexts.append(
            f"""
NOME FILE: {filename}

INTESTAZIONE CV:
{candidate_info}

CONTENUTO:
{document}
"""
        )

    final_context = "\n\n".join(contexts)

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""
DOMANDA UTENTE:
{user_question}

CONTESTO CV:
{final_context}

ISTRUZIONI:
- individua il candidato migliore
- spiega il motivo della scelta
- cita competenze rilevanti
- indica il nome del file
- non inventare informazioni
- se nessun candidato è adatto dillo chiaramente
"""

    messages = cl.user_session.get("messages", [])

    messages.append({
        "role": "user",
        "content": prompt
    })

    response_message = cl.Message(content="")

    await response_message.send()

    # =====================================================
    # STREAMING OLLAMA
    # =====================================================

    try:

        stream = ollama.chat(
            model="llama3.2",
            messages=messages,
            stream=True
        )

        full_response = ""

        for chunk in stream:

            token = chunk["message"]["content"]

            full_response += token

            await response_message.stream_token(token)

        messages.append({
            "role": "assistant",
            "content": full_response
        })

        cl.user_session.set("messages", messages)

        await response_message.update()

    except Exception as e:

        await cl.Message(
            content=f"Errore: {str(e)}"
        ).send()

# =========================================================
# CHAT END
# =========================================================

@cl.on_chat_end
async def on_chat_end():

    await cl.Message(
        content="Grazie per aver utilizzato HR Assistant 👋"
    ).send()