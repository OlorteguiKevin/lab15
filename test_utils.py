import utils
import chromadb

def test_create_chroma_collection():
    col = utils.create_chroma_collection("TestCol")
    assert col.name == "TestCol"

def test_embed_documents_custom():
    vectores = utils.embed_documents_custom(["hola mundo"])
    assert isinstance(vectores, list)
    assert isinstance(vectores[0], list)
    assert len(vectores[0]) > 0

def test_query_with_custom_embedding():
    docs = ["Luis es beneficiario de BECA 18"]
    col = utils.create_chroma_collection("TempTest")
    emb = utils.embed_documents_custom(docs)
    col.add(documents=docs, embeddings=emb, ids=["1"])
    res = utils.query_with_custom_embedding(col, "¿Quién tiene beca 18?")
    assert "Luis" in res
