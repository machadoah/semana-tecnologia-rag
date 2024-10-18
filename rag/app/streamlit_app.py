import streamlit as st
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from rag.ingestion import (
    default_path_index,
    create_vector_index,
    get_documents,
    get_vector_index,
)
from rag.querying import create_engine
from llama_index.core import Settings

# Configurações do modelo e embeddings
Settings.llm = Groq(model="llama-3.1-70b-versatile", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large", embed_batch_size=100
)


# Função para carregar ou criar o índice
@st.cache_resource
def load_or_create_index():
    if not default_path_index.exists():
        documents = get_documents()
        index = create_vector_index(documents=documents)
        index.storage_context.persist(persist_dir=default_path_index)
    else:
        index = get_vector_index(path_folder=default_path_index)
    return index


# Carrega o índice
index = load_or_create_index()
# Cria o motor de consulta
engine = create_engine(index=index)

# Interface com Streamlit
st.title("RAG sobre a Fatec")
st.write("Insira sua pergunta abaixo:")

# Caixa de texto para inserir a pergunta
pergunta = st.text_input("Pergunta:")

# Botão para enviar a pergunta
if st.button("Enviar"):
    if pergunta:
        # Realiza a consulta
        resposta = engine.query(pergunta)
        # Exibe a resposta
        st.write("### Resposta:")
        st.write(resposta.response)
    else:
        st.write("Por favor, insira uma pergunta válida.")

# Botão para sair
if st.button("Sair"):
    st.stop()
