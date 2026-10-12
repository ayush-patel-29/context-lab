import math

import pytest

from contextlab.metrics import evaluate, mrr_at_k, ndcg_at_k, rank_query, recall_at_k
from contextlab.stats import bootstrap_ci, paired_diff_ci

# Hand calculations (binary relevance, DCG = sum rel / log2(rank + 1)):
#  relevant at rank 2, 1 gold exists:  DCG = 1/log2(3) = 0.63093, ideal = 1      -> 0.63093
#  relevant at rank 1, 2 gold exist:   DCG = 1,         ideal = 1 + 1/log2(3)    -> 1 / 1.63093 = 0.61315


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([1, 0, 0], n_relevant=1) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], n_relevant=1) == 0.0


def test_ndcg_hand_computed():
    assert ndcg_at_k([0, 1, 0], n_relevant=1) == pytest.approx(0.6309297536)
    assert ndcg_at_k([1, 0, 0], n_relevant=2) == pytest.approx(0.6131471928)


def test_ndcg_respects_cutoff():
    labels = [0] * 10 + [1]  # relevant passage sits at rank 11
    assert ndcg_at_k(labels, n_relevant=1, k=10) == 0.0


def test_mrr():
    assert mrr_at_k([0, 0, 1]) == pytest.approx(1 / 3)
    assert mrr_at_k([0, 0, 0]) == 0.0


def test_recall_uses_total_gold_not_found_gold():
    assert recall_at_k([1, 0, 1], n_relevant=4, k=2) == pytest.approx(0.25)


def _row(q, p, label, bm25_rank, n_rel=1):
    return {"query_id": q, "passage_id": p, "label": label, "bm25_rank": bm25_rank, "n_relevant": n_rel}


def test_ties_fall_back_to_bm25_order():
    rows = [_row("q", "a", 0, 2), _row("q", "b", 1, 1)]
    scores = {("q", "a"): 0.9, ("q", "b"): 0.9}  # tie
    assert [r["passage_id"] for r in rank_query(rows, scores)] == ["b", "a"]


def test_evaluate_reranking_moves_gold_up():
    rows = [_row("q", "a", 0, 1), _row("q", "b", 1, 2)]
    bm25_like = {("q", "a"): 2.0, ("q", "b"): 1.0}
    reranked = {("q", "a"): 0.1, ("q", "b"): 0.9}
    assert evaluate(rows, bm25_like)["q"]["ndcg"] == pytest.approx(1 / math.log2(3))
    assert evaluate(rows, reranked)["q"]["ndcg"] == pytest.approx(1.0)


def test_bootstrap_ci():
    m, lo, hi = bootstrap_ci([0.5] * 20)
    assert m == lo == hi == 0.5
    assert bootstrap_ci([0, 1, 0, 1, 1]) == bootstrap_ci([0, 1, 0, 1, 1])  # seeded -> reproducible
    assert paired_diff_ci([0.2, 0.4], [0.2, 0.4]) == (0.0, 0.0, 0.0)