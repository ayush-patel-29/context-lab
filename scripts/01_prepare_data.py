import json
from pathlib import Path

from contextlab.data import load_scifact, make_split
from contextlab.retrieval import build_candidates, candidate_recall

OUT = Path("data/candidates.jsonl")
K = 30


def main():
    docs, queries, qrels = load_scifact()

    eligible = [q for q in queries if q in qrels]  # must have >= 1 relevant passage
    dev, test = make_split(eligible, n_total=100, n_dev=20, seed=42)
    split_of = {**{q: "dev" for q in dev}, **{q: "test" for q in test}}

    rows = build_candidates(docs, queries, qrels, split_of, k=K)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:  # utf-8 matters on Windows
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {OUT}  (expected {100 * K})")
    print(f"BM25 recall@{K}  all: {candidate_recall(rows, qrels):.3f}"
          f"  dev: {candidate_recall(rows, qrels, 'dev'):.3f}"
          f"  test: {candidate_recall(rows, qrels, 'test'):.3f}")


if __name__ == "__main__":
    main()