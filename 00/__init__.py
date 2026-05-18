import chainlit as cl

@cl.on_message
async def handler(message: cl.Message):
    response = f"Ciao, hai scritto: {message.content}"
    await cl.Message(response).send()