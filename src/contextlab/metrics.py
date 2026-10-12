import math
from collections import defaultdict


def dcg(ranked_labels, k):
    # rank r (1-based) is discounted by log2(r + 1); enumerate is 0-based, so +2
    return sum(lab / math.log2(i + 2) for i, lab in enumerate(ranked_labels[:k]))


def ndcg_at_k(ranked_labels, n_relevant, k=10):
    ideal = dcg([1] * min(n_relevant, k), k)  # best possible: all gold at the top
    return dcg(ranked_labels, k) / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_labels, k=10):
    for rank, lab in enumerate(ranked_labels[:k], start=1):
        if lab:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_labels, n_relevant, k=10):
    return sum(ranked_labels[:k]) / n_relevant if n_relevant else 0.0


def rank_query(rows, scores):
    """Order one query's candidate rows by score (high first).

    Ties are broken by BM25 rank, so a model that gives many identical
    scores (saturation) falls back to the BM25 order instead of arbitrary order.
    """
    return sorted(
        rows,
        key=lambda r: (-scores[(r["query_id"], r["passage_id"])], r["bm25_rank"]),
    )


def evaluate(rows, scores, k=10):
    """Per-query metrics. `scores` maps (query_id, passage_id) -> float."""
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["query_id"]].append(r)

    out = {}
    for qid, qrows in by_query.items():
        ranked = rank_query(qrows, scores)
        labels = [r["label"] for r in ranked]
        n_rel = qrows[0]["n_relevant"]
        out[qid] = {
            "ndcg": ndcg_at_k(labels, n_rel, k),
            "mrr": mrr_at_k(labels, k),
            "recall": recall_at_k(labels, n_rel, k),
        }
    return out