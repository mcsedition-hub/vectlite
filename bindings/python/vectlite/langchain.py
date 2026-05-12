"""LangChain integration for VectLite.

Usage::

    from langchain_openai import OpenAIEmbeddings
    from vectlite.langchain import VectLiteVectorStore

    store = VectLiteVectorStore(
        path="my.vdb",
        embedding=OpenAIEmbeddings(),
        dimension=1536,
    )
    store.add_texts(["hello world", "foo bar"])
    results = store.similarity_search("hello", k=3)

Requires ``langchain-core >= 0.2``.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, Sequence

import vectlite


def _import_langchain() -> tuple[Any, Any]:
    """Lazy-import langchain-core types so vectlite itself has no hard dep."""
    try:
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.vectorstores import VectorStore
    except ImportError as exc:
        raise ImportError(
            "langchain-core is required for VectLiteVectorStore. "
            "Install it with: pip install langchain-core"
        ) from exc
    return Document, Embeddings, VectorStore


class VectLiteVectorStore:
    """LangChain-compatible vector store backed by VectLite."""

    def __init__(
        self,
        path: str,
        embedding: Any,
        dimension: int | None = None,
        namespace: str | None = None,
        metric: str | None = None,
        text_key: str = "text",
    ) -> None:
        Document, Embeddings, VectorStore = _import_langchain()
        if not isinstance(embedding, Embeddings):
            raise TypeError(f"embedding must be a langchain Embeddings instance, got {type(embedding)}")
        self._db = vectlite.open(path, dimension=dimension, metric=metric)
        self._embedding = embedding
        self._namespace = namespace
        self._text_key = text_key
        self._Document = Document

    @property
    def db(self) -> vectlite.Database:
        """Access the underlying VectLite Database."""
        return self._db

    @property
    def embeddings(self) -> Any:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Returns a list of IDs for the added texts.
        """
        texts_list = list(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        vectors = self._embedding.embed_documents(texts_list)

        for doc_id, text, vector, meta in zip(ids, texts_list, vectors, metadatas):
            payload = dict(meta)
            payload[self._text_key] = text
            sparse = vectlite.sparse_terms(text)
            self._db.upsert(
                doc_id,
                list(vector),
                payload,
                namespace=self._namespace,
                sparse=sparse,
            )
        return ids

    def add_documents(
        self,
        documents: Sequence[Any],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add LangChain Document objects."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Return documents most similar to query."""
        docs_and_scores = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Any, float]]:
        """Return documents most similar to query, with scores."""
        vector = self._embedding.embed_query(query)
        sparse = vectlite.sparse_terms(query)
        results = self._db.search(
            list(vector),
            k=k,
            filter=filter,
            namespace=self._namespace,
            sparse=sparse,
            **kwargs,
        )
        docs_and_scores = []
        for result in results:
            metadata = dict(result.get("metadata", {}))
            text = metadata.pop(self._text_key, "")
            doc = self._Document(page_content=str(text), metadata=metadata)
            docs_and_scores.append((doc, result["score"]))
        return docs_and_scores

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Return documents most similar to embedding vector."""
        results = self._db.search(
            embedding,
            k=k,
            filter=filter,
            namespace=self._namespace,
            **kwargs,
        )
        docs = []
        for result in results:
            metadata = dict(result.get("metadata", {}))
            text = metadata.pop(self._text_key, "")
            docs.append(self._Document(page_content=str(text), metadata=metadata))
        return docs

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by IDs."""
        if ids is None:
            return None
        self._db.delete_many(ids, namespace=self._namespace)
        return True

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        path: str = ":memory:",
        dimension: int | None = None,
        namespace: str | None = None,
        metric: str | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> VectLiteVectorStore:
        """Create a VectLiteVectorStore from a list of texts."""
        if dimension is None:
            # Infer dimension from a single embedding
            sample = embedding.embed_query(texts[0] if texts else "")
            dimension = len(sample)
        store = cls(
            path=path,
            embedding=embedding,
            dimension=dimension,
            namespace=namespace,
            metric=metric,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def close(self) -> None:
        """Close the underlying database."""
        self._db.close()
