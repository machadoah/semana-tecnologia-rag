# Definir paths no projeto de forma mais robusta
from pathlib import Path
from typing import List
from llama_index.core import Document


# Estrutura para ler documentandos de um diretório
from llama_index.core.readers.file.base import SimpleDirectoryReader

# Constrói um índice vetorial dos documentos carregados
from llama_index.core import VectorStoreIndex

# Gera um indice dado uma contexto já criado
from llama_index.core import load_index_from_storage

# Estrutura de dados que armazena documentos e vetores
from llama_index.core.storage import StorageContext


default_path_index = Path("rag/vector_index")
default_path_data = Path("rag/data")


def get_documents(path_folder: Path = default_path_data) -> List[Document]:
    """ """
    return SimpleDirectoryReader(path_folder).load_data()


def create_vector_index(documents: List[Document]) -> VectorStoreIndex:
    """ """

    return VectorStoreIndex.from_documents(documents)


def get_vector_index(path_folder: Path = default_path_index) -> VectorStoreIndex:
    """ """
    storage_context = StorageContext.from_defaults(persist_dir=path_folder)
    return load_index_from_storage(storage_context=storage_context)
