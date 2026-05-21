import os
import chainlit as cl
import ollama

from db import CVDatabase
from config import DOCUMENTS_DIR, OLLAMA_MODEL
from document_prepocessor import DocumentProcessor
from semantic_chunking import SemanticChunking
from utils import leggi_prime_righe
from prompts import SYSTEM_PROMPT, build_prompt

db = CVDatabase()
added, updated, removed = DocumentProcessor.process_documents(db)
print(f"Document sync complete: {added} added, {updated} updated, {removed} removed")

@cl.action_callback("db_stats")
async def on_action(action: cl.Action):
    stats = db.get_stats()
    await cl.Message(content=stats).send()


@cl.action_callback("db_reindex")
async def on_action(action: cl.Action):
    added, updated, removed = DocumentProcessor.process_documents(db)
    message = f"DB reindicizzato con successo. Document sync complete: {added} added, {updated} updated, {removed} removed"
    await cl.Message(message).send()

@cl.on_chat_start
async def on_chat_start():

    actions = [
        cl.Action(
            name="db_stats",
            icon="mouse-pointer-click",
            payload={"value": "db_stats"},
            label="Statistiche Database",
        ),
        cl.Action(
            name="db_reindex",
            icon="mouse-pointer-click",
            payload={"value": "db_reindex"},
            label="Reindex Database",
        )
    ]

    await cl.Message(content="Informazioni del sistema:", actions=actions).send()

    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
    )

    await cl.Message(content="Ciao! Sono il tuo HR Assistant 🤖").send()


@cl.on_message
async def handle_message(message: cl.Message):

    user_question = message.content

    results = db.search(user_question, n_results=3)
    print(f"Risultati {results}")
    contexts = []

    for i in range(len(results["documents"][0])):

        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        filename = metadata["source"]

        filepath = os.path.join(DOCUMENTS_DIR, filename)
        candidate_info = leggi_prime_righe(filepath, 20)

        contexts.append(f"""
NOME FILE: {filename}

INTESTAZIONE CV:
{candidate_info}

CONTENUTO:
{document}
""")

    final_context = "\n\n".join(contexts)

    prompt = build_prompt(user_question, final_context)

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True
        )

        full_response = ""

        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            await response_message.stream_token(token)

        messages.append({"role": "assistant", "content": full_response})
        cl.user_session.set("messages", messages)

        await response_message.update()

    except Exception as e:
        await cl.Message(content=f"Errore: {str(e)}").send()


@cl.on_chat_end
async def on_chat_end():
    await cl.Message(content="Grazie per aver utilizzato HR Assistant 👋").send()