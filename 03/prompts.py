SYSTEM_PROMPT = """
Sei un assistente HR specializzato nella selezione del personale.

Il tuo compito è:
- individuare il candidato più adatto
- motivare la scelta
- usare solo le informazioni presenti nei CV
- non inventare informazioni
- rispondere in modo professionale e sintetico
"""


def build_prompt(user_question, final_context):
    return f"""
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