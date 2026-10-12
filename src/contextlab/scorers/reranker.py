from .base import Scorer


class RerankerScorer(Scorer):
    """Cross-encoder baseline: reads query and passage together, outputs a relevance score."""

    name = "reranker"
    batch_size = 32

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512):
        from sentence_transformers import CrossEncoder  # heavy import, only when used

        self.model = CrossEncoder(model_name, max_length=max_length)

    def score(self, rows):
        pairs = [(r["query_text"], r["passage_text"]) for r in rows]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]