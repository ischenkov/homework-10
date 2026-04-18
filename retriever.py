import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import Settings

settings = Settings()
INDEX_DIR = Path(__file__).parent / "index"

_embeddings = None
_vectorstore = None
_bm25_retriever = None
_ensemble_retriever = None
_reranker = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.api_key.get_secret_value(),
        )
    return _embeddings


def _load_chunks() -> list[Document]:
    chunks_path = INDEX_DIR / "chunks.pkl"
    if not chunks_path.exists():
        return []
    with open(chunks_path, "rb") as f:
        return pickle.load(f)


def _load_vectorstore():
    global _vectorstore
    if _vectorstore is None and INDEX_DIR.exists():
        embeddings = _get_embeddings()
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def _get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None:
        docs = _load_chunks()
        if not docs:
            return None
        _bm25_retriever = BM25Retriever.from_documents(docs)
        _bm25_retriever.k = 10
    return _bm25_retriever


def _get_ensemble_retriever():
    global _ensemble_retriever
    if _ensemble_retriever is None:
        vs = _load_vectorstore()
        bm25 = _get_bm25_retriever()
        if vs is None or bm25 is None:
            return None
        faiss_retriever = vs.as_retriever(search_kwargs={"k": 10})
        _ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[0.5, 0.5],
        )
    return _ensemble_retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def rerank_documents(query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
    if not documents:
        return []
    if len(documents) <= top_k:
        return documents
    reranker = _get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in doc_scores[:top_k]]


def search_knowledge_base(query: str, top_k: int = 5) -> list[Document]:
    ensemble = _get_ensemble_retriever()
    if ensemble is None:
        return []
    try:
        docs = ensemble.invoke(query)
        reranked = rerank_documents(query, docs, top_k=top_k)
        return reranked
    except Exception:
        return []
