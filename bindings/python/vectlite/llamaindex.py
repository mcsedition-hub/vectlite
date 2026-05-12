"""LlamaIndex integration for VectLite.

Usage::

    from llama_index.embeddings.openai import OpenAIEmbedding
    from vectlite.llamaindex import VectLiteVectorStore

    store = VectLiteVectorStore(
        path="my.vdb",
        dimension=1536,
    )

    # Use with a LlamaIndex StorageContext / VectorStoreIndex:
    from llama_index.core import StorageContext, VectorStoreIndex

    storage_ctx = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)
    query_engine = index.as_query_engine()

Requires ``llama-index-core >= 0.10``.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

import vectlite


def _import_llamaindex() -> tuple[Any, ...]:
    """Lazy-import llama-index-core types."""
    try:
        from llama_index.core.schema import BaseNode, TextNode
        from llama_index.core.vector_stores.types import (
            BasePydanticVectorStore,
            VectorStoreQuery,
            VectorStoreQueryResult,
        )
    except ImportError as exc:
        raise ImportError(
            "llama-index-core is required for VectLiteVectorStore. "
            "Install it with: pip install llama-index-core"
        ) from exc
    return BasePydanticVectorStore, VectorStoreQuery, VectorStoreQueryResult, BaseNode, TextNode


class VectLiteVectorStore:
    """LlamaIndex-compatible vector store backed by VectLite.

    This class follows the LlamaIndex VectorStore protocol without inheriting
    from BasePydanticVectorStore, keeping vectlite dependency-free at import
    time.  It implements the ``add``, ``delete``, and ``query`` methods that
    LlamaIndex expects.
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        path: str,
        dimension: int | None = None,
        namespace: str | None = None,
        metric: str | None = None,
        text_key: str = "text",
    ) -> None:
        self._db = vectlite.open(path, dimension=dimension, metric=metric)
        self._namespace = namespace
        self._text_key = text_key

    @property
    def db(self) -> vectlite.Database:
        """Access the underlying VectLite Database."""
        return self._db

    @property
    def client(self) -> vectlite.Database:
        """LlamaIndex convention — alias for the underlying client."""
        return self._db

    def add(
        self,
        nodes: Sequence[Any],
        **kwargs: Any,
    ) -> list[str]:
        """Add LlamaIndex nodes (TextNode / BaseNode) to the store.

        Each node must have a non-None ``embedding`` set.
        Returns a list of node IDs.
        """
        ids: list[str] = []
        for node in nodes:
            node_id = node.node_id or str(uuid.uuid4())
            embedding = node.get_embedding()
            if embedding is None:
                raise ValueError(f"Node {node_id} has no embedding set")

            metadata = dict(node.metadata or {})
            text = getattr(node, "text", "") or ""
            metadata[self._text_key] = text

            sparse = vectlite.sparse_terms(text) if text else {}
            self._db.upsert(
                node_id,
                list(embedding),
                metadata,
                namespace=self._namespace,
                sparse=sparse,
            )
            ids.append(node_id)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete a node by its ID."""
        self._db.delete(ref_doc_id, namespace=self._namespace)

    def query(
        self,
        query: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a VectorStoreQuery and return a VectorStoreQueryResult."""
        _, _, VectorStoreQueryResult, _, TextNode = _import_llamaindex()

        query_embedding = query.query_embedding
        k = query.similarity_top_k or 10
        filters = None  # TODO: map query.filters to vectlite filter format

        sparse: dict[str, float] = {}
        if query.query_str:
            sparse = vectlite.sparse_terms(query.query_str)

        results = self._db.search(
            list(query_embedding) if query_embedding is not None else None,
            k=k,
            filter=filters,
            namespace=self._namespace,
            sparse=sparse if sparse else None,
        )

        nodes: list[Any] = []
        similarities: list[float] = []
        ids: list[str] = []

        for result in results:
            metadata = dict(result.get("metadata", {}))
            text = metadata.pop(self._text_key, "")
            node = TextNode(
                text=str(text),
                metadata=metadata,
                id_=result["id"],
            )
            nodes.append(node)
            similarities.append(result["score"])
            ids.append(result["id"])

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def close(self) -> None:
        """Close the underlying database."""
        self._db.close()
