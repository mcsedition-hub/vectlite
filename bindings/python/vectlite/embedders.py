"""Built-in embedding providers for vectlite.

Each factory returns a callable ``(text: str) -> list[float]`` that can be
passed directly to :func:`vectlite.upsert_text` and :func:`vectlite.search_text`::

    from vectlite import embedders

    embed = embedders.openai("text-embedding-3-small")
    vectlite.upsert_text(db, "doc1", "Hello world", embed)
    results = vectlite.search_text(db, "greeting", embed)

All providers lazy-import their SDK so vectlite itself has no hard
dependency on any of them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def openai(
    model: str = "text-embedding-3-small",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
    timeout: float | None = None,
) -> Callable[[str], list[float]]:
    """Create an embedder using the OpenAI Embeddings API.

    Requires ``openai`` package::

        pip install openai

    Args:
        model: Model name (e.g. ``"text-embedding-3-small"``,
            ``"text-embedding-3-large"``, ``"text-embedding-ada-002"``).
        api_key: API key.  Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Override base URL (for Azure, proxies, etc.).
        dimensions: Optional output dimension override (embedding-3 models).
        timeout: Request timeout in seconds.
    """
    try:
        import openai as _openai  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "openai is required for the OpenAI embedder. "
            "Install with: pip install openai"
        ) from exc

    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout

    client = _openai.OpenAI(**kwargs)

    def embed(text: str) -> list[float]:
        params: dict[str, Any] = {"input": text, "model": model}
        if dimensions is not None:
            params["dimensions"] = dimensions
        response = client.embeddings.create(**params)
        return list(response.data[0].embedding)

    embed.__name__ = f"openai:{model}"  # type: ignore[attr-defined]
    embed.dimension = dimensions  # type: ignore[attr-defined]
    embed.model = model  # type: ignore[attr-defined]
    return embed


def cohere(
    model: str = "embed-english-v3.0",
    *,
    api_key: str | None = None,
    input_type: str = "search_document",
    truncate: str | None = None,
) -> Callable[[str], list[float]]:
    """Create an embedder using the Cohere Embed API.

    Requires ``cohere`` package::

        pip install cohere

    Args:
        model: Model name (e.g. ``"embed-english-v3.0"``,
            ``"embed-multilingual-v3.0"``).
        api_key: API key.  Falls back to ``CO_API_KEY`` env var.
        input_type: One of ``"search_document"``, ``"search_query"``,
            ``"classification"``, ``"clustering"``.
        truncate: Truncation strategy (``"NONE"``, ``"START"``, ``"END"``).
    """
    try:
        import cohere as _cohere  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "cohere is required for the Cohere embedder. "
            "Install with: pip install cohere"
        ) from exc

    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    client = _cohere.ClientV2(**kwargs)

    def embed(text: str) -> list[float]:
        params: dict[str, Any] = {
            "texts": [text],
            "model": model,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        if truncate is not None:
            params["truncate"] = truncate
        response = client.embed(**params)
        return list(response.embeddings.float_[0])

    embed.__name__ = f"cohere:{model}"  # type: ignore[attr-defined]
    embed.model = model  # type: ignore[attr-defined]
    return embed


def voyage(
    model: str = "voyage-3",
    *,
    api_key: str | None = None,
    input_type: str | None = None,
    truncation: bool = True,
) -> Callable[[str], list[float]]:
    """Create an embedder using the Voyage AI API.

    Requires ``voyageai`` package::

        pip install voyageai

    Args:
        model: Model name (e.g. ``"voyage-3"``, ``"voyage-3-lite"``,
            ``"voyage-code-3"``).
        api_key: API key.  Falls back to ``VOYAGE_API_KEY`` env var.
        input_type: ``"document"`` or ``"query"`` (optional).
        truncation: Whether to truncate long inputs.
    """
    try:
        import voyageai as _voyageai  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "voyageai is required for the Voyage embedder. "
            "Install with: pip install voyageai"
        ) from exc

    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    client = _voyageai.Client(**kwargs)

    def embed(text: str) -> list[float]:
        params: dict[str, Any] = {
            "texts": [text],
            "model": model,
            "truncation": truncation,
        }
        if input_type is not None:
            params["input_type"] = input_type
        response = client.embed(**params)
        return list(response.embeddings[0])

    embed.__name__ = f"voyage:{model}"  # type: ignore[attr-defined]
    embed.model = model  # type: ignore[attr-defined]
    return embed


def fastembed(
    model: str = "BAAI/bge-small-en-v1.5",
    *,
    cache_dir: str | None = None,
    threads: int | None = None,
    max_length: int = 512,
) -> Callable[[str], list[float]]:
    """Create a local embedder using FastEmbed (ONNX Runtime).

    Requires ``fastembed`` package::

        pip install fastembed

    This runs entirely locally with no API calls.

    Args:
        model: Model name (e.g. ``"BAAI/bge-small-en-v1.5"``,
            ``"BAAI/bge-base-en-v1.5"``, ``"sentence-transformers/all-MiniLM-L6-v2"``).
        cache_dir: Directory for downloaded model files.
        threads: Number of threads for ONNX runtime.
        max_length: Maximum token length.
    """
    try:
        from fastembed import TextEmbedding  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "fastembed is required for the local FastEmbed embedder. "
            "Install with: pip install fastembed"
        ) from exc

    kwargs: dict[str, Any] = {"model_name": model, "max_length": max_length}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if threads is not None:
        kwargs["threads"] = threads
    embedding_model = TextEmbedding(**kwargs)

    def embed(text: str) -> list[float]:
        # fastembed.embed() returns a generator of numpy arrays
        vectors = list(embedding_model.embed([text]))
        return [float(x) for x in vectors[0]]

    embed.__name__ = f"fastembed:{model}"  # type: ignore[attr-defined]
    embed.model = model  # type: ignore[attr-defined]
    return embed


def sentence_transformer(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    device: str | None = None,
    normalize: bool = True,
) -> Callable[[str], list[float]]:
    """Create a local embedder using SentenceTransformers (PyTorch).

    Requires ``sentence-transformers`` package::

        pip install sentence-transformers

    Args:
        model: HuggingFace model name or local path.
        device: Device to run on (e.g. ``"cpu"``, ``"cuda"``).
        normalize: Whether to L2-normalize embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for the SentenceTransformer embedder. "
            "Install with: pip install sentence-transformers"
        ) from exc

    st_model = SentenceTransformer(model, device=device)

    def embed(text: str) -> list[float]:
        vector = st_model.encode(text, normalize_embeddings=normalize, convert_to_numpy=True)
        return [float(x) for x in vector]

    embed.__name__ = f"sentence_transformer:{model}"  # type: ignore[attr-defined]
    embed.model = model  # type: ignore[attr-defined]
    return embed


def ollama(
    model: str = "nomic-embed-text",
    *,
    host: str | None = None,
) -> Callable[[str], list[float]]:
    """Create an embedder using a local Ollama server.

    Requires ``ollama`` package::

        pip install ollama

    Make sure the Ollama server is running and the model is pulled::

        ollama pull nomic-embed-text

    Args:
        model: Model name (e.g. ``"nomic-embed-text"``,
            ``"mxbai-embed-large"``).
        host: Ollama server URL (default: ``http://localhost:11434``).
    """
    try:
        import ollama as _ollama  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "ollama is required for the Ollama embedder. "
            "Install with: pip install ollama"
        ) from exc

    kwargs: dict[str, Any] = {}
    if host is not None:
        kwargs["host"] = host
    client = _ollama.Client(**kwargs)

    def embed(text: str) -> list[float]:
        response = client.embed(model=model, input=text)
        return list(response["embeddings"][0])

    embed.__name__ = f"ollama:{model}"  # type: ignore[attr-defined]
    embed.model = model  # type: ignore[attr-defined]
    return embed


__all__ = [
    "cohere",
    "fastembed",
    "ollama",
    "openai",
    "sentence_transformer",
    "voyage",
]
