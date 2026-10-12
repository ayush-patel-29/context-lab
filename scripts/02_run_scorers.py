import argparse
import json
import time
from pathlib import Path

from contextlab.jsonl import read_jsonl
from contextlab.scorers.bm25 import BM25Scorer
from contextlab.scorers.reranker import RerankerScorer
from contextlab.scorers.jev import JevScorer

CANDIDATES = Path("data/candidates.jsonl")
RESULTS = Path("results")


def make_scorer(name):
    if name == "bm25":
        return BM25Scorer()
    
    if name == "reranker":
        return RerankerScorer()
    
    if name.startswith("jev_"):  # e.g. jev_v1_n30 = wording v1, 30 passages per request
        _, wording, n = name.split("_")
        return JevScorer(wording=wording, pairs_per_call=int(n.removeprefix("n")))
    
    raise SystemExit(f"unknown scorer: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = ap.parse_args()

    rows = read_jsonl(CANDIDATES)
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"scores_{args.scorer}.jsonl"
    done = {(r["query_id"], r["passage_id"]) for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if (r["query_id"], r["passage_id"]) not in done]
    print(f"{args.scorer}: {len(rows)} pairs requested, {len(done)} cached, {len(todo)} to score")

    scorer = make_scorer(args.scorer)
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), scorer.batch_size):
            batch = todo[i : i + scorer.batch_size]
            t0 = time.perf_counter()
            scores = scorer.score(batch)
            per_pair = (time.perf_counter() - t0) / len(batch)
            assert len(scores) == len(batch), "scorer must return one score per row"
            for r, s in zip(batch, scores):
                f.write(json.dumps({
                    "query_id": r["query_id"], "passage_id": r["passage_id"],
                    "score": float(s), "latency_s": per_pair,
                }) + "\n")
            f.flush()  # keep progress even if a later batch crashes
            print(f"  {min(i + scorer.batch_size, len(todo))}/{len(todo)}")


if __name__ == "__main__":
    main()