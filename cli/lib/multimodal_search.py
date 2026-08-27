import os
from typing import Any

from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    format_search_result,
    load_movies,
)
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(
        self, documents: list[dict] | None = None, model_name: str = "clip-ViT-B-32"
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> NDArray[Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        embedding = self.model.encode([image])  # type: ignore[arg-type]
        return embedding[0]  # type: ignore[return-value]

    def search_with_image(
        self, image_path: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict[str, Any]]:
        image_embedding = self.embed_image(image_path)

        similarities: list[tuple[int, float]] = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in similarities[:limit]:
            doc = self.documents[idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return results


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(
    image_path: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict[str, Any]]:
    documents = load_movies()
    search = MultimodalSearch(documents)
    return search.search_with_image(image_path, limit)
