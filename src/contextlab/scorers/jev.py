from typesafe_sdk import TypeSafeClient
import random
import os
from dotenv import load_dotenv
from .base import Scorer
load_dotenv()

# Question wordings. Try at most two on the DEV split, freeze one, then run TEST.
# Changing a wording = a new wording key, so cached scores are never mixed up.
WORDINGS = {
    "v1": "Does the abstract in {p} contain evidence that supports or refutes the claim in {q}?",
    "v2": "Would a scientist checking the claim in {q} find {p} useful as evidence?",
}


class JevScorer(Scorer):
    """Score = Jev's P(yes) for a yes/no (Noul) question about each passage.

    pairs_per_call = how many passages share ONE request (one Noul question each).
    30 = one request per query (cheap, fast); 1 = one request per pair.
    """

    batch_size = 30

    def __init__(self, wording="v1", pairs_per_call=30, client=None):
        self.wording = wording
        self.pairs_per_call = pairs_per_call
        self.name = f"jev_{wording}_n{pairs_per_call}"  # also the cache file name
        if client is None:

            client = TypeSafeClient()
        self.client = client

    def score(self, rows):
        out = [0.0] * len(rows)
        by_query = {}
        for i, r in enumerate(rows):
            by_query.setdefault(r["query_id"], []).append(i)

        for idxs in by_query.values():
            for start in range(0, len(idxs), self.pairs_per_call):
                chunk = idxs[start : start + self.pairs_per_call]
                probs = self._ask(rows[chunk[0]]["query_text"], [rows[i]["passage_text"] for i in chunk])
                for i, p in zip(chunk, probs):
                    out[i] = p
        return out

    def _ask(self, query, passages):
        from typesafe_sdk import Noul

        # Shuffle passage order (seeded by the query) so BM25 order can't leak into
        # Jev's judgments through position; then map the answers back.
        order = list(range(len(passages)))
        random.Random(query).shuffle(order)
        shuffled = [passages[j] for j in order]

        template = WORDINGS[self.wording]
        questions = {
            f"p{k}": Noul(instructions=template.format(q="`claim`", p=f"`passages[{k}]`"))
            for k in range(len(shuffled))
        }
        resp = self.client.system_one(
            state={"claim": query, "passages": shuffled}, questions=questions
        )

        probs = [0.0] * len(passages)
        for k, j in enumerate(order):
            probs[j] = float(resp.answers[f"p{k}"].noul)
        return probs