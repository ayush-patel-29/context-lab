from contextlab.data import make_split
from contextlab.retrieval import build_candidates, candidate_recall, tokenize

IDS = [str(i) for i in range(300)]


def test_split_sizes_and_disjoint():
    dev, test = make_split(IDS)
    assert len(dev) == 20 and len(test) == 80
    assert not set(dev) & set(test)


def test_split_is_reproducible_and_order_independent():
    assert make_split(IDS) == make_split(IDS)
    assert make_split(IDS) == make_split(list(reversed(IDS)))


def test_tokenize_same_for_query_and_passage():
    assert tokenize("Hello, World-19!") == ["hello", "world", "19"]


DOCS = {
    "d1": "cats purr and sleep",
    "d2": "dogs bark loudly",
    "d3": "stock markets fell sharply",
    "d4": "birds sing at dawn",
}
QUERIES = {"q1": "why do cats purr", "q2": "dogs barking"}
QRELS = {"q1": {"d1"}, "q2": {"d2", "d3"}}  # d3 is unreachable by q2 on purpose


def test_candidates_shape_labels_and_no_leakage():
    rows = build_candidates(DOCS, QUERIES, QRELS, {"q1": "dev", "q2": "test"}, k=2)
    assert len(rows) == 4
    q1 = [r for r in rows if r["query_id"] == "q1"]
    assert q1[0]["passage_id"] == "d1" and q1[0]["label"] == 1 and q1[0]["bm25_rank"] == 1
    # q2's second gold (d3) was not retrieved -> must NOT be injected
    assert "d3" not in {r["passage_id"] for r in rows if r["query_id"] == "q2"}


def test_recall_counts_missed_gold():
    rows = build_candidates(DOCS, QUERIES, QRELS, {"q1": "dev", "q2": "test"}, k=2)
    assert candidate_recall(rows, QRELS, "dev") == 1.0
    assert candidate_recall(rows, QRELS, "test") == 0.5  # found d2, missed d3