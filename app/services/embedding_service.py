"""
Generates ONE embedding for a full document (resume or job description).
No chunking anywhere in this system.

Default: local sentence-transformers model, so the project runs without any
external embedding API key. Swap `_model` for a hosted client (e.g. Voyage AI,
which Anthropic recommends for embeddings) if you want a hosted model instead
- the rest of the codebase only depends on `embed_text()`'s signature.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.core.exceptions import EmbeddingGenerationError

settings = get_settings()


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    if not text or not text.strip():
        raise EmbeddingGenerationError("Cannot embed empty text.")
    try:
        vector = _model().encode(text, normalize_embeddings=True)
        return vector.tolist()
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingGenerationError(str(exc)) from exc
