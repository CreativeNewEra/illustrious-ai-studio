from functools import lru_cache
from typing import Any


def compute_embeddings(prompt: str) -> Any:
    """Compute prompt embeddings (placeholder implementation)."""
    # NOTE: Real embedding computation would use a model.
    # Here we simply return a hashed representation for caching purposes.
    return hash(prompt)


@lru_cache(maxsize=100)
def get_model_embeddings(prompt: str) -> Any:
    """Cache prompt embeddings for faster regeneration."""
    return compute_embeddings(prompt)
