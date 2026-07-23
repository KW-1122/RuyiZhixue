from __future__ import annotations

import os
from pathlib import Path


class VectorStore:
    """FAISS cosine index backed by a local Chinese BGE embedding model."""

    def __init__(self, documents: list, root: Path):
        self.enabled = os.getenv("ENABLE_VECTOR_RAG", "false").lower() == "true"
        self.documents = documents
        self.model = None
        self.index = None
        if not self.enabled:
            return
        model_path = Path(os.getenv("EMBEDDING_MODEL_PATH", root / "models" / "bge-small-zh-v1.5"))
        if not model_path.exists():
            self.enabled = False
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(model_path), local_files_only=True)
            texts = [f"{d.title}\n{d.content}" for d in documents]
            vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors.astype("float32"))
        except Exception:
            self.enabled = False

    def scores(self, query: str, top_k: int = 12) -> dict[int, float]:
        if not self.enabled or self.model is None or self.index is None:
            return {}
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        scores, indices = self.index.search(vector, min(top_k, len(self.documents)))
        return {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}
