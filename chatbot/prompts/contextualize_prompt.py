from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt que define a forma com a qual a LLM precisa operar 
# durante a interação com o usuário no escopo da recuperação 
# de informações a partir do RAG. 
# 
# Por meio dele, permite à LLM persistir o que já foi recuperado 
# e a instrui responder conforme o esperado, sem precisar buscar 
# a mesma informação no banco de dados vetorial novamente. 

context_q_system_prompt = """\
Dado um histórico de bate-papo e a última pergunta do usuário 
que pode fazer referência ao contexto no histórico do bate-papo,
formule uma pergunta independente que possa ser entendida
sem o histórico de bate-papo. NÃO responda à pergunta,
apenas reformule-o se necessário e devolva-o como está.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_q_system_prompt), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}")
    ]
)