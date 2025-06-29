from typing import List, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def create_chroma_collection(name: str) -> chromadb.api.models.Collection.Collection:
    """
    Crea una colección ChromaDB local.

    Args:
        name (str): Nombre de la colección

    Returns:
        Collection: Objeto colección Chroma
    """
    client = chromadb.Client(Settings())
    return client.create_collection(name=name)


def add_documents_to_collection(
    collection: chromadb.api.models.Collection.Collection,
    docs: List[str]
) -> None:
    """
    Agrega documentos a una colección ChromaDB.

    Args:
        collection (Collection): Colección destino
        docs (List[str]): Lista de textos a almacenar
    """
    collection.add(documents=docs, ids=[f"doc{i}" for i in range(len(docs))])


def query_collection(
    collection: chromadb.api.models.Collection.Collection,
    query: str,
    n_results: int = 1
) -> str:
    """
    Consulta documentos relevantes desde la base vectorial.

    Args:
        collection (Collection): Colección ChromaDB
        query (str): Texto de consulta
        n_results (int): Cantidad de resultados deseados

    Returns:
        str: Documento más relevante
    """
    result = collection.query(query_texts=[query], n_results=n_results)
    return result['documents'][0][0]


def embed_documents_custom(
    docs: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> List[List[float]]:
    """
    Genera embeddings personalizados usando SentenceTransformer.

    Args:
        docs (List[str]): Lista de textos
        model_name (str): Nombre del modelo embedding

    Returns:
        List[List[float]]: Lista de vectores embeddings
    """
    model = SentenceTransformer(model_name)
    return model.encode(docs).tolist()


def query_with_custom_embedding(
    collection: chromadb.api.models.Collection.Collection,
    query: str,
    model_name: str = 'all-MiniLM-L6-v2'
) -> str:
    """
    Realiza consulta usando embeddings personalizados.

    Args:
        collection (Collection): Colección destino
        query (str): Pregunta
        model_name (str): Modelo embedding a usar

    Returns:
        str: Documento relevante
    """
    model = SentenceTransformer(model_name)
    vector = model.encode([query]).tolist()
    result = collection.query(query_embeddings=vector, n_results=1)
    return result['documents'][0][0]
