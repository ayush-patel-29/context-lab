import argparse
from pathlib import Path

from contextlab.jsonl import read_jsonl
from contextlab.metrics import evaluate
from contextlab.stats import bootstrap_ci, paired_diff_ci

CANDIDATES = Path("data/candidates.jsonl")
RESULTS = Path("results")


def fmt(ci):
    m, lo, hi = ci
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--baseline", default="bm25")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    rows = [r for r in read_jsonl(CANDIDATES) if r["split"] == args.split]
    qids = sorted({r["query_id"] for r in rows})

    results = {}
    for path in sorted(RESULTS.glob("scores_*.jsonl")):
        name = path.stem.removeprefix("scores_")
        recs = read_jsonl(path)
        scores = {(r["query_id"], r["passage_id"]): r["score"] for r in recs}
        lat = {(r["query_id"], r["passage_id"]): r["latency_s"] for r in recs}
        if any((r["query_id"], r["passage_id"]) not in scores for r in rows):
            print(f"skipping {name}: not all {args.split} pairs are scored yet")
            continue
        per_q = evaluate(rows, scores, k=args.k)
        ms = 1000 * sum(lat[(r["query_id"], r["passage_id"])] for r in rows) / len(rows)
        results[name] = (per_q, ms)
        
    if not results:
       raise SystemExit("No complete scorer results found. Run 02_run_scorers.py first.")

    k = args.k
    base = results.get(args.baseline)
    lines = [
        f"Split: {args.split}  ({len(qids)} queries, top-30 BM25 candidates; mean [95% bootstrap CI])",
        "",
        f"| scorer | nDCG@{k} | MRR@{k} | recall@{k} | ΔnDCG vs {args.baseline} | ms/pair |",
        "|---|---|---|---|---|---|",
    ]
    for name, (per_q, ms) in results.items():
        col = lambda key: [per_q[q][key] for q in qids]
        if base and name != args.baseline:
            delta = fmt(paired_diff_ci(col("ndcg"), [base[0][q]["ndcg"] for q in qids]))
        else:
            delta = "-"
        lines.append(
            f"| {name} | {fmt(bootstrap_ci(col('ndcg')))} | {fmt(bootstrap_ci(col('mrr')))} "
            f"| {fmt(bootstrap_ci(col('recall')))} | {delta} | {ms:.1f} |"
        )

    text = "\n".join(lines)
    print(text)
    (RESULTS / "table.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()