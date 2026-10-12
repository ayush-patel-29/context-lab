import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase, keep letters/digits. Used for BOTH passages and queries."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, docs):
        self.doc_ids = list(docs)
        self.bm25 = BM25Okapi([tokenize(docs[i]) for i in self.doc_ids])

    def top_k(self, query_text, k):
        scores = self.bm25.get_scores(tokenize(query_text))
        order = np.argsort(-scores, kind="stable")[:k]  # stable: ties keep corpus order
        return [(self.doc_ids[i], float(scores[i])) for i in order]


def build_candidates(docs, queries, qrels, split_of, k=30, index=None):
    """One row per (query, candidate passage).

    split_of: {query_id: "dev" | "test"}
    The gold passage is NEVER added if BM25 missed it (that would leak labels).
    """
    index = index or BM25Index(docs)
    rows = []
    for qid, split in split_of.items():
        for rank, (did, score) in enumerate(index.top_k(queries[qid], k), start=1):
            rows.append(
                {
                    "query_id": qid,
                    "split": split,
                    "query_text": queries[qid],
                    "passage_id": did,
                    "passage_text": docs[did],
                    "bm25_score": score,
                    "bm25_rank": rank,
                    "label": int(did in qrels[qid]),
                    "n_relevant": len(qrels[qid]),  # total gold passages, needed for nDCG/recall
                }
            )
    return rows


def candidate_recall(rows, qrels, split=None):
    """Average over queries of: relevant passages found / relevant passages that exist."""
    found = {}
    for r in rows:
        if split and r["split"] != split:
            continue
        found.setdefault(r["query_id"], set())
        if r["label"]:
            found[r["query_id"]].add(r["passage_id"])
    per_query = [len(found[q]) / len(qrels[q]) for q in found]
    return sum(per_query) / len(per_query)