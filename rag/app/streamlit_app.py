# Importa a biblioteca Streamlit para criar interfaces web interativas
import streamlit as st

# Importa a classe para embeddings usando o modelo Hugging Face
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Importa a classe do modelo Groq para uso com LlamaIndex
from llama_index.llms.groq import Groq

# Importa funções de ingestão de dados e criação de índices
from rag.ingestion import (
    default_path_index,
    create_vector_index,
    get_documents,
    get_vector_index,
)

# Importa a função para criar um motor de busca no índice vetorial
from rag.querying import create_engine

# Configurações globais do LlamaIndex
from llama_index.core import Settings

# Configurações do modelo e embeddings
Settings.llm = Groq(model="llama-3.1-70b-versatile", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large", embed_batch_size=100
)


# Função para carregar ou criar o índice
@st.cache_resource
def load_or_create_index():
    """
    Carrega ou cria um índice vetorial a partir dos documentos.

    Se o índice não existir no diretório especificado, ele será criado
    com base nos documentos carregados. Caso contrário, o índice existente
    será carregado para uso.

    Returns:
        VectorStoreIndex: O índice vetorial carregado ou criado.
    """
    if not default_path_index.exists():
        documents = get_documents()
        index = create_vector_index(documents=documents)
        index.storage_context.persist(persist_dir=default_path_index)
    else:
        index = get_vector_index(path_folder=default_path_index)
    return index


# Carrega o índice vetorial, seja criando um novo ou carregando um existente
index = load_or_create_index()

# Cria o motor de busca para o índice vetorial
engine = create_engine(index=index)

# Interface com Streamlit
st.title("RAG sobre a Fatec")
st.write("Insira sua pergunta abaixo:")

# Caixa de texto para inserir a pergunta
pergunta = st.text_input("Pergunta:")

# Botão para enviar a pergunta
if st.button("Enviar"):
    if pergunta:
        # Realiza a consulta no motor de busca e obtém a resposta
        resposta = engine.query(pergunta)
        # Exibe a resposta na interface
        st.write("### Resposta:")
        st.write(resposta.response)
    else:
        # Mensagem de erro caso a pergunta seja inválida
        st.write("Por favor, insira uma pergunta válida.")

# Botão para encerrar a execução do aplicativo
if st.button("Sair"):
    st.stop()
