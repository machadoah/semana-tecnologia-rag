# Estrutura do motor de busca para informações no índice
from llama_index.core.base.base_query_engine import BaseQueryEngine

from llama_index.core import VectorStoreIndex


def create_engine(index: VectorStoreIndex) -> BaseQueryEngine:
    return index.as_query_engine()
