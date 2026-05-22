import os
import chainlit as cl
from document_processor import DocumentProcessor
from database import Database
from config import Config
from utils import LLMHelper
import shutil

db = Database()

# Process documents
dp = DocumentProcessor()
added, updated, removed = dp.process_documents(db)
print(f"Document sync complete: {added} added, {updated} updated, {removed} removed")

@cl.action_callback("db_remove")
async def on_action(action: cl.Action):
    db.remove_collection()
    await cl.Message("Db eliminato correttamente. È necessario lanciare il reindex").send()

@cl.action_callback("db_stats")
async def on_action(action: cl.Action):
    print(action.payload)
    db_info = db.get_stats()
    print(db_info)
    response = await LLMHelper.get_db_stats(db_info)
    await cl.Message(response).send()


@cl.action_callback("db_reindex")
async def on_action(action: cl.Action):
    added, updated, removed = dp.process_documents(db)
    message = f"DB reindicizzato con successo. Document sync complete: {added} added, {updated} updated, {removed} removed"
    await cl.Message(message).send()


@cl.on_chat_start
async def start():

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
        ),
        cl.Action(
            name="db_remove",
            icon="mouse-pointer-click",
            payload={"value": "db_remove"},
            label="Rimozione Database",
        ),
    ]

    await cl.Message(content="Informazioni del sistema:", actions=actions).send()

    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": """
                    Sei un assistente specializzato nel mondo HR, rispondi in modo professionale, sintetico e pragmatico.
                    Il tuo ruolo è individuare il candidato ideale rispetto alle richieste dell'utente.
                """,
            }
        ],
    )


@cl.on_message
async def handle_message(message: cl.Message):
    # gestisco l'upload dei file
    # gestisco l'upload dei file
    if message.elements:
        print("message.elements", message.elements)
        await cl.Message(content="Caricamento e indicizzazione documenti").send()

        files = [
            file
            for file in message.elements
            if file.name.lower().endswith(tuple(DocumentProcessor.SUPPORTED_EXTENSIONS))
        ]

        # Elaborazione parallela dei file
        if files:
            results = [await _file_upload(file) for file in files]
        else:
            results = ["Nessun file caricato"]
        # ["File /blabla.pdf caricato e indicizzato con successo.", "File /blabla2.pdf caricato e indicizzato con successo."]


        result_message = "\n".join(results)
        await cl.Message(content=result_message).send()
        await cl.Message(content=f"Caricati {len(files)} file").send()


    user_question = message.content
    results = db.query(user_question)

    filename = results["metadatas"][0][0]["source"]
    context_lines = DocumentProcessor.read_first_lines(
        os.path.join(Config.DOCUMENTS_DIR, filename), 200
    )

    context = f"CONTESTO: nome file {results['metadatas'][0][0]['source']} ecco il paragrafo piu' significativo: {results['documents'][0][0]}"

    candidate_name = await LLMHelper.get_candidate_name(context_lines)

    prompt = LLMHelper.create_prompt(context, user_question, candidate_name)

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = LLMHelper.chat(messages)

        for chunk in stream:
            await response_message.stream_token(
                str(chunk.choices[0].delta.content or "")
            )

        messages.append({"role": "assistant", "content": response_message.content})
        await response_message.update()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await cl.Message(content=error_message).send()
        print(error_message)

    cl.user_session.set("messages", messages)


async def _process_and_index_file(file_path: str, file_name: str) -> str:
    """Elabora e indicizza il documento, restituendo un messaggio di esito."""
    documents, metadatas, ids = dp.process_single_document(file_path)
    if documents:
        db.add_documents(documents, metadatas, ids)
        return f"File '{file_name}' caricato e indicizzato con successo."
    return f"Errore nel processare il file '{file_name}'."


async def _file_upload(file) -> str:
    # Salva il file nella directory specificata
    file_name = file.name
    src_file_path = file.path
    dst_file_path = os.path.join(Config.DOCUMENTS_DIR, file_name)
    os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
    shutil.move(src_file_path, dst_file_path)

    # processa e indicizza il file
    return await _process_and_index_file(dst_file_path, file_name)
