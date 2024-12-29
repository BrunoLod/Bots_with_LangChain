from pprint import pprint
from typing import List

from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from prompts.contextualize_prompt import contextualize_q_prompt
from prompts.system_prompt import prompt


class ChatBot():
    """
    A chatbot class designed to provide conversational AI capabilities 
    with retrieval-augmented generation (RAG) functionality.

    Attributes:
        llm (ChatGroq): The language model used for generating responses.
        retriever_prompt (ChatPromptTemplate): Template for constructing retriever prompts.
        system_prompt (ChatPromptTemplate): Template for system initialization prompts.
        embedding (HuggingFaceEmbeddings): Embedding model used for vectorization.
        documents (List[Document]): The input list of documents to be processed.
        store (dict): A dictionary to store session histories by session ID.
    """
    
    def __init__(
        self,
        llm: ChatGroq,
        retriever_prompt: ChatPromptTemplate,
        system_prompt: ChatPromptTemplate,
        embedding: HuggingFaceEmbeddings,
        documents: List[Document]
    ) -> None:
        """
        Initializes the ChatBot with the specified components.

        Args:
            llm (ChatGroq): The language model for conversation.
            retriever_prompt (ChatPromptTemplate): Prompt template for retriever.
            system_prompt (ChatPromptTemplate): Prompt template for system initialization.
            embedding (HuggingFaceEmbeddings): Embedding model for document vectorization.
            documents (List[Document]): List of documents to be loaded and processed.
        """
        self.llm              = llm
        self.retriever_prompt = retriever_prompt
        self.system_prompt    = system_prompt
        self.embedding        = embedding
        self.documents        = documents
        self.store            = {}

    def loader(self) -> List[Document]:
        """
        Loads documents using a web-based loader.

        Returns:
            List[Document]: A list of loaded documents.
        """
        loader = WebBaseLoader(self.documents)
        return loader.load()
    
    def splitter(self) -> List[Document]:
        """
        Splits documents into smaller chunks for processing.

        Returns:
            List[Document]: A list of split document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size         = 500,
            chunk_overlap      = 50,
            length_function    = len,
            separators         = ["", " ", ".", "\n", "\n\n"],
            is_separator_regex = False
        )
        return text_splitter.split_documents(self.loader())
    
    def retriver(self) -> InMemoryVectorStore:
        """
        Creates an in-memory vector store for document retrieval.

        Returns:
            InMemoryVectorStore: A retriever for document vectors.
        """
        vectorstore = InMemoryVectorStore.from_documents(
            self.splitter(),
            self.embedding
        )
        return vectorstore.as_retriever()
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves or initializes chat history for a given session.

        Args:
            session_id (str): Unique identifier for the session.

        Returns:
            BaseChatMessageHistory: Chat history for the session.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
            
    def run(self, query: str) -> str:
        """
        Executes the chatbot pipeline for a given query.

        Args:
            query (str): The user query to process.

        Returns:
            str: The chatbot's response to the query.
        """
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriver(), 
            self.retriever_prompt
        )

        qa_chain = create_stuff_documents_chain(
            self.llm, 
            self.system_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain
        ) 

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key   = "input",
            history_messages_key = "chat_history",
            output_messages_key  = "answer"
        )

        return conversational_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": "dsm_v"}
            }
        )["answer"]

# For local test:
if __name__ == "__main__":

    llm = ChatGroq(
      model = "llama3-70b-8192", 
      temperature = 0.5, 
      api_key = "your_api_key"  
    ) 

    system_prompt = prompt
    retriever_prompt = contextualize_q_prompt

    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2"
    )

    url = "https://en-m-wikipedia-org.translate.goog/wiki/Dark_wave?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=tc"

    rag_bot = ChatBot(
        llm              = llm, 
        retriever_prompt = retriever_prompt, 
        system_prompt    = prompt,
        embedding        = embedding, 
        documents        = url
    )

    print("Olá! Eu sou o RagBot, digite a sua pergunta ou escreva 'sair' para encerrar a nossa conversa.")
    while True: 
        user_input = input("Mensagem: ")

        if user_input.lower() == "sair":
            print("Encerrando a conversa")
            break

        try: 
            response = rag_bot.run(user_input)
            print(f"Rag Bot: {response}")
        except Exception as e:
            print(f"Erro ao processar a resposta {e}")