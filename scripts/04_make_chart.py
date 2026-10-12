import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contextlab.jsonl import read_jsonl
from contextlab.metrics import evaluate
from contextlab.stats import bootstrap_ci, paired_diff_ci

CANDIDATES = Path("data/candidates.jsonl")
RESULTS = Path("results")

# (file name without "scores_", label, color)
SERIES = [
    ("bm25", "BM25\n(no rerank)", "#9AA0A6"),
    ("reranker", "Cross-encoder\nreranker", "#4C78A8"),
    ("jev_v1_n30", "Jev", "#E4572E"),
]


def draw(labels, means, errs, colors, n_queries, split, k, note, out):
    """errs: list of (lower_err, upper_err) per bar."""
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
    x = range(len(means))
    ax.bar(x, means, color=colors, width=0.55)
    ax.errorbar(x, means, yerr=list(zip(*errs)), fmt="none", ecolor="#222", capsize=6, lw=1.6)
    for i, m in enumerate(means):
        ax.text(i, m + errs[i][1] + 0.015, f"{m:.3f}", ha="center", fontsize=14, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(f"nDCG@{k}  (higher is better)", fontsize=12)
    ax.set_title(f"Reranking SciFact candidates: Jev vs a cross-encoder reranker ({n_queries} {split} queries)",
                 fontsize=13, loc="left", fontweight="bold", pad=30)
    if note:
        ax.text(0, 1.03, note, transform=ax.transAxes, fontsize=11, color="#333")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.01,
             "Error bars: 95% bootstrap CI. Latency: Jev = hosted API, 30 passages per request; "
             "reranker = MiniLM-L6 on a laptop CPU, so not like-for-like hardware.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(CANDIDATES) if r["split"] == args.split]
    qids = sorted({r["query_id"] for r in rows})

    labels, means, errs, colors, per_query = [], [], [], [], {}
    for name, label, color in SERIES:
        path = RESULTS / f"scores_{name}.jsonl"
        if not path.exists():
            continue
        recs = read_jsonl(path)
        scores = {(r["query_id"], r["passage_id"]): r["score"] for r in recs}
        lat = {(r["query_id"], r["passage_id"]): r["latency_s"] for r in recs}
        if any((r["query_id"], r["passage_id"]) not in scores for r in rows):
            continue
        per_q = evaluate(rows, scores, k=args.k)
        vals = [per_q[q]["ndcg"] for q in qids]
        m, lo, hi = bootstrap_ci(vals)
        ms = 1000 * sum(lat[(r["query_id"], r["passage_id"])] for r in rows) / len(rows)
        labels.append(f"{label}\n{ms:.0f} ms/pair" if ms >= 1 else label)
        means.append(m)
        errs.append((m - lo, hi - m))
        colors.append(color)
        per_query[name] = vals

    note = ""
    if "jev_v1_n30" in per_query and "reranker" in per_query:
        d, lo, hi = paired_diff_ci(per_query["jev_v1_n30"], per_query["reranker"])
        verdict = "no clear difference" if lo <= 0 <= hi else "clear difference"
        note = f"Jev minus reranker: {d:+.3f} nDCG@{args.k} (95% CI {lo:+.3f} to {hi:+.3f}), {verdict}"

    draw(labels, means, errs, colors, len(qids), args.split, args.k, note, RESULTS / f"ndcg_{args.split}.png")


if __name__ == "__main__":
    main()