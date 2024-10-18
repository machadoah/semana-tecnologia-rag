# Estrutura do motor de busca para consultas no índice vetorial
from llama_index.core.base.base_query_engine import BaseQueryEngine

# Classe para criar e gerenciar um índice vetorial
from llama_index.core import VectorStoreIndex


def create_engine(index: VectorStoreIndex) -> BaseQueryEngine:
    """
    Cria um motor de busca para realizar consultas no índice vetorial.

    Esta função converte o índice vetorial fornecido em um motor de busca
    que pode ser utilizado para executar consultas e recuperar informações
    dos documentos indexados.

    Args:
        index (VectorStoreIndex): O índice vetorial que será usado para criar o motor de busca.

    Returns:
        BaseQueryEngine: Um objeto do tipo `BaseQueryEngine` que permite realizar consultas no índice.
    """
    return index.as_query_engine()
