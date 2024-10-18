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

Settings.llm = Groq(model="llama-3.1-70b-versatile", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large", embed_batch_size=100
)

stop_works: list = ["quit", "exit", "0", "sair", "parar", "stop"]


def main():
    # etapa de ingestão
    if not default_path_index.exists():
        # pega os documentos que servirão como conhecimento
        documents = get_documents()

        # gera a base de vetores com os documentos
        index = create_vector_index(documents=documents)

        # persiste o indice
        index.storage_context.persist(persist_dir=default_path_index)

    if default_path_index.exists():
        index = get_vector_index(path_folder=default_path_index)

    # etapa de recuperação/retriever
    engine = create_engine(index=index)

    while True:
        pergunta = input("Insira sua pergunta: ")

        if pergunta.lower() in stop_works:
            break

        # Realiza uma consulta usando o motor de consulta
        r = engine.query(pergunta)

        print(r)


if __name__ == "__main__":
    main()
