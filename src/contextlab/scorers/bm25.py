from .base import Scorer


class BM25Scorer(Scorer):
    """Baseline: keep the first-stage BM25 score (no reranking)."""

    name = "bm25"
    batch_size = 1000

    def score(self, rows):
        return [r["bm25_score"] for r in rows]