import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
load_dotenv(ROOT / ".env")

METHODS = ("bm25", "reranker", "jev_v1_n30")
VALID_LABELS = {"SUPPORT", "CONTRADICT", "MIXED", "NO_EVIDENCE"}
TIMING_VERSION = "online_query_v1"
TIMING_FIELDS = (
    "retrieval_latency_s",
    "scoring_latency_s",
    "context_prepare_latency_s",
    "generation_latency_s",
    "end_to_end_latency_s",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def query_sort_key(qid: str):
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def mean_of(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def format_mean(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def format_ci(values) -> str:
    if values is None:
        return "—"
    point, lower, upper = values
    return f"{point:.3f} [{lower:.3f}, {upper:.3f}]"


def class_metrics(y_true, y_pred, labels):
    """Return per-class precision, recall, F1, and support."""
    result = {}

    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        support = sum(t == label for t in y_true)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return result


def metric_values(y_true, y_pred, labels):
    per_class = class_metrics(y_true, y_pred, labels)

    return {
        "accuracy": float(
            np.mean(np.asarray(y_true) == np.asarray(y_pred))
        ),
        "macro_f1": float(
            np.mean([per_class[label]["f1"] for label in labels])
        ),
    }


def bootstrap_analysis(y_true, predictions, labels, n_boot: int, seed: int):
    """Paired query-level bootstrap CIs and paired differences versus BM25."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(n_boot, n))

    point = {
        method: metric_values(y_true, predictions[method], labels)
        for method in METHODS
    }

    boot_metrics = {
        method: {name: [] for name in ("accuracy", "macro_f1")}
        for method in METHODS
    }

    boot_deltas = {
        method: {name: [] for name in ("accuracy", "macro_f1")}
        for method in METHODS
        if method != "bm25"
    }

    for indices in samples:
        y_sample = [y_true[i] for i in indices]
        sampled = {}

        for method in METHODS:
            p_sample = [predictions[method][i] for i in indices]
            sampled[method] = metric_values(y_sample, p_sample, labels)

            for name, value in sampled[method].items():
                boot_metrics[method][name].append(value)

        for method in METHODS:
            if method == "bm25":
                continue

            for name in ("accuracy", "macro_f1"):
                boot_deltas[method][name].append(
                    sampled[method][name] - sampled["bm25"][name]
                )

    intervals = {}
    paired_differences = {}

    for method in METHODS:
        intervals[method] = {}
        paired_differences[method] = {}

        for name in ("accuracy", "macro_f1"):
            values = np.asarray(boot_metrics[method][name], dtype=float)

            intervals[method][name] = [
                point[method][name],
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]

            if method == "bm25":
                paired_differences[method][name] = None
            else:
                deltas = np.asarray(boot_deltas[method][name], dtype=float)
                observed = point[method][name] - point["bm25"][name]

                paired_differences[method][name] = [
                    float(observed),
                    float(np.quantile(deltas, 0.025)),
                    float(np.quantile(deltas, 0.975)),
                ]

    return point, intervals, paired_differences


def context_metrics(row: dict, n_relevant: int) -> dict:
    passages = row.get("context_passages") or []
    relevant_count = sum(
        bool(passage.get("qrel_relevant")) for passage in passages
    )

    return {
        "passage_count": len(passages),
        "context_chars": row.get("context_chars", 0),
        "relevant_passages": relevant_count,
        "context_precision": (
            relevant_count / len(passages) if passages else 0.0
        ),
        "context_recall": (
            relevant_count / n_relevant if n_relevant else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RAG answer quality, context, tokens, and latency."
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--model", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    model = args.model or os.getenv(
        "GEMINI_MODEL", "gemini-3.5-flash-lite"
    )
    model_slug = slug(model)

    candidate_path = DATA / "candidates.jsonl"
    result_path = (
        RESULTS / f"rag_gemini_{model_slug}_{args.split}.jsonl"
    )

    if not candidate_path.exists():
        raise SystemExit(f"Missing candidate file: {candidate_path}")
    if not result_path.exists():
        raise SystemExit(f"Missing RAG results file: {result_path}")

    candidates = [
        row
        for row in read_jsonl(candidate_path)
        if row.get("split") == args.split
    ]

    first_candidate = {}
    for row in candidates:
        qid = str(row["query_id"])
        first_candidate.setdefault(qid, row)

    query_ids = sorted(first_candidate, key=query_sort_key)

    if not query_ids:
        raise SystemExit(f"No candidates found for split {args.split}")

    n_relevant = {
        qid: int(first_candidate[qid]["n_relevant"])
        for qid in query_ids
    }

    records = read_jsonl(result_path)
    by_key = {}

    for row in records:
        qid = str(row.get("query_id", ""))
        method = row.get("method")
        key = (qid, method)

        if method not in METHODS:
            raise SystemExit(
                f"Unexpected method in RAG results: {method!r}"
            )
        if key in by_key:
            raise SystemExit(f"Duplicate result record: {key}")
        if row.get("status") != "ok":
            raise SystemExit(f"Unsuccessful result record: {key}")
        if row.get("timing_version") != TIMING_VERSION:
            raise SystemExit(
                f"Legacy or missing timing fields for {key}. "
                "Rerun scripts/08_run_rag_experiment.py with the "
                "corrected runner."
            )
        if row.get("split") != args.split or row.get("model") != model:
            raise SystemExit(f"Split/model mismatch in record {key}")

        for field in TIMING_FIELDS:
            value = row.get(field)

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value < 0
            ):
                raise SystemExit(
                    f"Missing or invalid {field} in record {key}"
                )

        by_key[key] = row

    missing = [
        (qid, method)
        for qid in query_ids
        for method in METHODS
        if (qid, method) not in by_key
    ]

    if missing:
        print(
            f"Incomplete result set: {len(missing)} "
            "query/method records missing."
        )
        for key in missing[:20]:
            print(f"  Missing: {key}")

        raise SystemExit(
            "Analysis stopped to avoid reporting metrics from a partial split. "
            "Run scripts/08_run_rag_experiment.py to finish the split."
        )

    y_true = []

    for qid in query_ids:
        rows = [by_key[(qid, method)] for method in METHODS]
        expected = {row.get("expected_output") for row in rows}
        inputs = {row.get("input") for row in rows}

        if len(expected) != 1:
            raise SystemExit(
                f"Expected labels differ by method for query {qid}"
            )
        if len(inputs) != 1:
            raise SystemExit(
                f"Claim text differs by method for query {qid}"
            )

        label = expected.pop()

        if label not in VALID_LABELS:
            raise SystemExit(
                f"Invalid expected label for query {qid}: {label}"
            )

        y_true.append(label)

    predictions = {
        method: [
            by_key[(qid, method)].get("predicted_label", "INVALID")
            for qid in query_ids
        ]
        for method in METHODS
    }

    # Use a fixed label set so macro-F1 is comparable across splits.
    labels = sorted(VALID_LABELS)

    point, intervals, paired_differences = bootstrap_analysis(
        y_true,
        predictions,
        labels,
        args.bootstrap_samples,
        args.seed,
    )

    resources = {}
    per_class = {}
    per_query = []

    for method in METHODS:
        rows = [by_key[(qid, method)] for qid in query_ids]

        contexts = [
            context_metrics(row, n_relevant[str(row["query_id"])])
            for row in rows
        ]
        usages = [row.get("usage") or {} for row in rows]

        prompt_tokens = [usage.get("prompt_tokens") for usage in usages]
        completion_tokens = [
            usage.get("completion_tokens") for usage in usages
        ]
        total_tokens = [usage.get("total_tokens") for usage in usages]

        resources[method] = {
            "claims": len(rows),
            "mean_retrieval_latency_s": mean_of(
                [row["retrieval_latency_s"] for row in rows]
            ),
            "mean_scoring_latency_s": mean_of(
                [row["scoring_latency_s"] for row in rows]
            ),
            "mean_retrieval_plus_scoring_s": mean_of([
                row["retrieval_latency_s"] + row["scoring_latency_s"]
                for row in rows
            ]),
            "mean_context_prepare_latency_s": mean_of(
                [row["context_prepare_latency_s"] for row in rows]
            ),
            "mean_generation_latency_s": mean_of(
                [row["generation_latency_s"] for row in rows]
            ),
            "mean_end_to_end_latency_s": mean_of(
                [row["end_to_end_latency_s"] for row in rows]
            ),
            "mean_prompt_tokens": mean_of(prompt_tokens),
            "mean_completion_tokens": mean_of(completion_tokens),
            "mean_total_tokens": mean_of(total_tokens),
            "sum_prompt_tokens": sum(
                value for value in prompt_tokens
                if isinstance(value, int) and not isinstance(value, bool)
            ),
            "sum_completion_tokens": sum(
                value for value in completion_tokens
                if isinstance(value, int) and not isinstance(value, bool)
            ),
            "sum_total_tokens": sum(
                value for value in total_tokens
                if isinstance(value, int) and not isinstance(value, bool)
            ),
            "mean_context_chars": mean_of(
                [context["context_chars"] for context in contexts]
            ),
            "mean_passages": mean_of(
                [context["passage_count"] for context in contexts]
            ),
            "mean_context_precision": mean_of(
                [context["context_precision"] for context in contexts]
            ),
            "mean_context_recall": mean_of(
                [context["context_recall"] for context in contexts]
            ),
            "invalid_predictions": sum(
                prediction not in VALID_LABELS
                for prediction in predictions[method]
            ),
            "token_usage_records_missing": sum(
                not all(
                    isinstance(usage.get(key), int)
                    and not isinstance(usage.get(key), bool)
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    )
                )
                for usage in usages
            ),
        }

        per_class[method] = class_metrics(
            y_true, predictions[method], labels
        )

    for index, qid in enumerate(query_ids):
        item = {
            "query_id": qid,
            "expected": y_true[index],
        }

        for method in METHODS:
            row = by_key[(qid, method)]

            item[f"{method}_prediction"] = row.get("predicted_label")
            item[f"{method}_correct"] = (
                row.get("predicted_label") == y_true[index]
            )
            item[f"{method}_scoring_latency_s"] = (
                row["scoring_latency_s"]
            )
            item[f"{method}_end_to_end_latency_s"] = (
                row["end_to_end_latency_s"]
            )
            item[f"{method}_total_tokens"] = (
                row.get("usage") or {}
            ).get("total_tokens")

        per_query.append(item)

    report = {
        "experiment": "retrieved_context_claim_verification_v1",
        "timing_version": TIMING_VERSION,
        "split": args.split,
        "model": model,
        "claims": len(query_ids),
        "methods": list(METHODS),
        "labels_present_in_reference": labels,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "metrics": point,
        "metric_confidence_intervals_95": intervals,
        "paired_differences_vs_bm25_95": paired_differences,
        "per_class_metrics": per_class,
        "resources": resources,
        "per_query": per_query,
        "caveats": [
            "Intervals are query-level percentile bootstrap intervals.",
            "Small development sets yield preliminary estimates.",
            "Latency excludes one-time index/model initialization.",
            "End-to-end latency is the sum of measured sequential stages.",
            "Context precision and recall use cited-document relevance judgments.",
            "API cost is not calculated in this report.",
        ],
    }

    json_path = (
        RESULTS / f"rag_analysis_{model_slug}_{args.split}.json"
    )
    md_path = (
        RESULTS / f"rag_analysis_{model_slug}_{args.split}.md"
    )

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# RAG experiment analysis",
        "",
        f"- Split: `{args.split}`",
        f"- Model: `{model}`",
        f"- Claims: **{len(query_ids)}**",
        f"- Bootstrap samples: {args.bootstrap_samples}",
        f"- Reference labels represented: {', '.join(labels)}",
        "",
        "Intervals are 95% query-level percentile intervals. Method differences "
        "versus BM25 are paired on the same resampled claims.",
        "",
        "## Answer quality, latency, and context",
        "",
        "| Method | Accuracy (95% CI) | Macro-F1 (95% CI) | "
        "Δ Accuracy vs BM25 (95% CI) | Δ Macro-F1 vs BM25 (95% CI) | "
        "Retrieval + scoring (s) | Scoring only (s) | Gemini generation (s) | "
        "End-to-end (s) | Mean prompt tokens | Mean total tokens | "
        "Context precision | Context recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for method in METHODS:
        resource = resources[method]

        lines.append(
            f"| {method} "
            f"| {format_ci(intervals[method]['accuracy'])} "
            f"| {format_ci(intervals[method]['macro_f1'])} "
            f"| {format_ci(paired_differences[method]['accuracy'])} "
            f"| {format_ci(paired_differences[method]['macro_f1'])} "
            f"| {format_mean(resource['mean_retrieval_plus_scoring_s'])} "
            f"| {format_mean(resource['mean_scoring_latency_s'])} "
            f"| {format_mean(resource['mean_generation_latency_s'])} "
            f"| {format_mean(resource['mean_end_to_end_latency_s'])} "
            f"| {format_mean(resource['mean_prompt_tokens'], 1)} "
            f"| {format_mean(resource['mean_total_tokens'], 1)} "
            f"| {format_mean(resource['mean_context_precision'], 3)} "
            f"| {format_mean(resource['mean_context_recall'], 3)} |"
        )

    lines.extend([
        "",
        "## Per-class metrics",
        "",
        "| Method | Class | Support | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|",
    ])

    for method in METHODS:
        for label in labels:
            values = per_class[method][label]
            lines.append(
                f"| {method} | {label} | {values['support']} "
                f"| {values['precision']:.3f} "
                f"| {values['recall']:.3f} "
                f"| {values['f1']:.3f} |"
            )

    lines.extend([
        "",
        "## Resource totals",
        "",
        "| Method | Prompt tokens | Completion tokens | Total tokens | "
        "Mean context chars | Mean passages | Invalid predictions | "
        "Missing token records |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for method in METHODS:
        resource = resources[method]
        lines.append(
            f"| {method} "
            f"| {resource['sum_prompt_tokens']} "
            f"| {resource['sum_completion_tokens']} "
            f"| {resource['sum_total_tokens']} "
            f"| {format_mean(resource['mean_context_chars'], 1)} "
            f"| {format_mean(resource['mean_passages'], 2)} "
            f"| {resource['invalid_predictions']} "
            f"| {resource['token_usage_records_missing']} |"
        )

    lines.extend([
        "",
        "## Per-query predictions",
        "",
        "| Query | Expected | BM25 | Reranker | JEV |",
        "|---|---|---|---|---|",
    ])

    for row in per_query:
        lines.append(
            f"| {row['query_id']} | {row['expected']} "
            f"| {row['bm25_prediction']} "
            f"| {row['reranker_prediction']} "
            f"| {row['jev_v1_n30_prediction']} |"
        )

    lines.extend([
        "",
        "## Interpretation cautions",
        "",
        "- A small development set is not sufficient to establish method superiority.",
        "- BM25 has no separate reranking stage; its search time is included "
        "in retrieval + scoring.",
        "- End-to-end latency excludes one-time index and model initialization.",
        "- Generation latency can fluctuate independently of the ranking method.",
        "- Context precision/recall measure document relevance, not semantic faithfulness.",
        "- This report records token counts, not API cost.",
        "",
        f"Machine-readable report: `{json_path.name}`",
    ])

    markdown = "\n".join(lines) + "\n"
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"Full report written to: {md_path}")
    print(f"Machine-readable report written to: {json_path}")


if __name__ == "__main__":
    main()
