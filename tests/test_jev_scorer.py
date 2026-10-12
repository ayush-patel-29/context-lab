from types import SimpleNamespace
from contextlab.scorers.jev import JevScorer

class FakeClient:
    """Pretends to be Jev: P(yes) = len(passage)/100, so we know the right answer."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append(len(state["passages"]))
        answers = {
            qid: SimpleNamespace(noul=len(state["passages"][int(qid[1:])]) / 100)
            for qid in questions
        }
        return SimpleNamespace(answers=answers)


def _rows(n, qid="q1"):
    return [
        {"query_id": qid, "query_text": "claim " + qid, "passage_text": "x" * (i + 1)}
        for i in range(n)
    ]


def test_scores_map_back_to_original_order_despite_shuffle():
    scorer = JevScorer(pairs_per_call=30, client=FakeClient())
    scores = scorer.score(_rows(10))
    assert scores == [(i + 1) / 100 for i in range(10)]


def test_chunking_and_grouping_by_query():
    fake = FakeClient()
    scorer = JevScorer(pairs_per_call=4, client=fake)
    scores = scorer.score(_rows(10, "q1") + _rows(3, "q2"))
    assert fake.calls == [4, 4, 2, 3]  # q1 split 4+4+2, q2 in one call
    assert len(scores) == 13


def test_name_encodes_wording_and_batching():
    assert JevScorer(wording="v2", pairs_per_call=1, client=FakeClient()).name == "jev_v2_n1"