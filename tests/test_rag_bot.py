"""
Teste unitário para o rag_bot.
"""

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from chatbot.rag_bot import RagBot


@patch('langchain_community.document_loaders.WebBaseLoader')
def test_loader(MockEWebBaseLoader):

    doc1 = "Where love is the law, youtopia"
    doc2 = "Alone with everyone we wanna dull the hurt"

    mock_loader = MockEWebBaseLoader.return_value 
    
    mock_loader.load.return_value = [doc1, doc2]

    llm = MagicMock()
    prompt = MagicMock()
    embedding = MagicMock()
    document = MagicMock()

    rag_bot = RagBot(
        llm=llm, 
        prompt=prompt, 
        embedding=embedding, 
        documents=document
    )

    rag_bot.documents = ["http://darkwave.com"]

    result = rag_bot.loader()

    # Asserts
    mock_loader.load.assert_called_once()
    unittest.TestCase.assertEqual(result, [doc1, doc2])

    # Testes para os outros métodos (...)