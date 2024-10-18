# Definir paths no projeto de forma mais robusta
from pathlib import Path

# Estrutura de lista
from typing import List

# Estrutura de documento do LlamaIndex
from llama_index.core import Document

# Estrutura para ler documentandos de um diretório
from llama_index.core.readers.file.base import SimpleDirectoryReader

# Constrói um índice vetorial dos documentos carregados
from llama_index.core import VectorStoreIndex

# Gera um indice dado uma contexto já criado
from llama_index.core import load_index_from_storage

# Estrutura de dados que armazena documentos e vetores
from llama_index.core.storage import StorageContext

# Caminho padrão para o índice de vetores e os dados utilizados no projeto
default_path_index = Path("rag/vector_index")
default_path_data = Path("rag/data")


def get_documents(path_folder: Path = default_path_data) -> List[Document]:
    """
    Carrega documentos de um diretório especificado.

    Esta função utiliza o `SimpleDirectoryReader` para ler e carregar
    todos os documentos presentes no diretório especificado.

    Args:
        path_folder (Path): O caminho para o diretório contendo os documentos.
                            O padrão é `default_path_data`.

    Returns:
        List[Document]: Uma lista de objetos `Document` representando os documentos carregados.
    """
    
    return SimpleDirectoryReader(path_folder).load_data()


def create_vector_index(documents: List[Document]) -> VectorStoreIndex:
    """
    Cria um índice vetorial a partir de uma lista de documentos.

    Esta função gera um índice vetorial, que pode ser utilizado para
    operações de busca e recuperação de informações com base nos documentos fornecidos.

    Args:
        documents (List[Document]): A lista de documentos a ser usada para criar o índice vetorial.

    Returns:
        VectorStoreIndex: Um índice vetorial criado a partir dos documentos fornecidos.
    """

    return VectorStoreIndex.from_documents(documents)


def get_vector_index(path_folder: Path = default_path_index) -> VectorStoreIndex:
    """
    Carrega um índice vetorial persistido a partir de um diretório.

    Esta função carrega um índice vetorial previamente salvo, utilizando
    o `StorageContext` para acessar os dados de armazenamento persistido.

    Args:
        path_folder (Path): O caminho para o diretório contendo o índice vetorial persistido.
                            O padrão é `default_path_index`.

    Returns:
        VectorStoreIndex: O índice vetorial carregado a partir do armazenamento persistido.
    """
    
    storage_context = StorageContext.from_defaults(persist_dir=path_folder)
    return load_index_from_storage(storage_context=storage_context)
