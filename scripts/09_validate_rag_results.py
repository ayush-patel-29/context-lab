import argparse

import json

import math

import sys

from collections import defaultdict

from pathlib import Path

from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

METHODS = {"bm25", "reranker", "jev_v1_n30"}

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

    """Read JSONL records and report malformed rows clearly."""

    records = []

    with path.open(encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():

                continue

            try:

                records.append(json.loads(line))

            except json.JSONDecodeError as exc:

                raise ValueError(

                    f"Invalid JSON at line {line_number}: {exc}"

                ) from exc

    return records

def natural_key(value: str):

    """Sort numeric query IDs numerically, then other IDs alphabetically."""

    return (0, int(value)) if value.isdigit() else (1, value)

def get_result_path(model: str, split: str) -> Path:

    """Build the same result filename used by scripts/08_run_rag_experiment.py."""

    import re

    model_slug = re.sub(

        r"[^a-zA-Z0-9]+",

        "_",

        model,

    ).strip("_").lower()

    return RESULTS / f"rag_gemini_{model_slug}_{split}.jsonl"

def validate_nonnegative_number(value) -> bool:

    """True if value is a finite, nonnegative number."""

    return (

        isinstance(value, (int, float))

        and not isinstance(value, bool)

        and math.isfinite(value)

        and value >= 0

    )

def main() -> int:

    parser = argparse.ArgumentParser(

        description="Validate RAG records and online latency measurements."

    )

    parser.add_argument(

        "--split",

        choices=("dev", "test"),

        default="dev",

    )

    parser.add_argument(

        "--model",

        default="gemini-3.5-flash-lite",

        help="Gemini model name used to create the result file.",

    )

    args = parser.parse_args()

    result_path = get_result_path(args.model, args.split)

    manifest_path = RESULTS / (

        f"rag_timing_manifest_"

        f"{result_path.name.removeprefix('rag_gemini_').removesuffix('.jsonl')}"

        f".json"

    )

    if not result_path.exists():

        print(f"ERROR: Result file not found: {result_path}")

        return 1

    try:

        records = read_jsonl(result_path)

    except (OSError, ValueError) as exc:

        print(f"ERROR: Could not read result file: {exc}")

        return 1

    if not records:

        print("ERROR: Result file is empty.")

        return 1

    errors = []

    warnings = []

    seen = set()

    by_query = defaultdict(dict)

    method_times = defaultdict(lambda: defaultdict(list))

    token_totals = defaultdict(

        lambda: {

            "prompt": 0,

            "completion": 0,

            "total": 0,

            "count": 0,

            "missing_usage": 0,

        }

    )

    for index, row in enumerate(records, start=1):

        qid = str(row.get("query_id", ""))

        method = row.get("method")

        key = (qid, method)

        if not qid:

            errors.append(f"Row {index}: missing query_id")

        if method not in METHODS:

            errors.append(f"Row {index}: unknown method {method!r}")

        if key in seen:

            errors.append(f"Row {index}: duplicate query/method pair {key}")

        seen.add(key)

        if row.get("timing_version") != TIMING_VERSION:

            errors.append(

                f"Row {index}: timing_version is missing or outdated. "

                "Run the corrected experiment with --overwrite."

            )

        if row.get("status") != "ok":

            errors.append(f"Row {index}: status is not 'ok'")

        if row.get("split") != args.split:

            errors.append(

                f"Row {index}: expected split {args.split!r}, "

                f"got {row.get('split')!r}"

            )

        if row.get("model") != args.model:

            errors.append(

                f"Row {index}: expected model {args.model!r}, "

                f"got {row.get('model')!r}"

            )

        if row.get("predicted_label") not in VALID_LABELS:
            warnings.append(
                f"Row {index}: invalid model prediction "
                f"{row.get('predicted_label')!r}; retain it and count it as incorrect"
            )

        if row.get("expected_output") not in VALID_LABELS:

            errors.append(f"Row {index}: invalid expected label")

        # Validate all timing fields.

        timing_values = {}

        for field in TIMING_FIELDS:

            value = row.get(field)

            if not validate_nonnegative_number(value):

                errors.append(

                    f"Row {index}: missing or invalid timing field {field}"

                )

            else:

                timing_values[field] = float(value)

                if method in METHODS:

                    method_times[method][field].append(float(value))

        # End-to-end latency should equal the sum of sequential stages.

        required_stages = (

            "retrieval_latency_s",

            "scoring_latency_s",

            "context_prepare_latency_s",

            "generation_latency_s",

        )

        if all(field in timing_values for field in required_stages) and (

            "end_to_end_latency_s" in timing_values

        ):

            stage_sum = sum(

                timing_values[field] for field in required_stages

            )

            recorded_total = timing_values["end_to_end_latency_s"]

            if abs(stage_sum - recorded_total) > 0.002:

                errors.append(

                    f"Row {index}: end-to-end latency does not match "

                    f"the sum of stages ({recorded_total:.4f}s vs "

                    f"{stage_sum:.4f}s)"

                )

            # The legacy latency field should equal Gemini generation time.

            legacy_latency = row.get("latency_s")

            if (

                not validate_nonnegative_number(legacy_latency)

                or abs(

                    float(legacy_latency)

                    - timing_values["generation_latency_s"]

                ) > 0.002

            ):

                errors.append(

                    f"Row {index}: latency_s does not match "

                    "generation_latency_s"

                )

            # BM25 scoring is included in retrieval, not a second ranking call.

            if (

                method == "bm25"

                and timing_values["scoring_latency_s"] > 0.002

            ):

                errors.append(

                    f"Row {index}: BM25 has a nonzero separate scoring stage. "

                    "Its retrieval and ranking should be represented by "

                    "retrieval_latency_s."

                )

        # Validate context passage metadata.

        passages = row.get("context_passages")

        context_ids = row.get("context_passage_ids")

        retrieved_ids = row.get("retrieved_candidate_ids")

        if not isinstance(passages, list):

            errors.append(f"Row {index}: context_passages must be a list")

            passages = []

        if not isinstance(context_ids, list):

            errors.append(f"Row {index}: context_passage_ids must be a list")

            context_ids = []

        if not isinstance(retrieved_ids, list):

            errors.append(

                f"Row {index}: retrieved_candidate_ids must be a list"

            )

            retrieved_ids = []

        passage_ids_from_context = [

            str(passage.get("passage_id", ""))

            for passage in passages

        ]

        context_ids = [str(value) for value in context_ids]

        retrieved_ids = [str(value) for value in retrieved_ids]

        if passage_ids_from_context != context_ids:

            errors.append(

                f"Row {index}: context_passage_ids do not match "

                "the IDs in context_passages"

            )

        if len(set(context_ids)) != len(context_ids):

            errors.append(f"Row {index}: duplicate passage in selected context")

        if len(set(retrieved_ids)) != len(retrieved_ids):

            errors.append(f"Row {index}: duplicate BM25 candidate ID")

        if not set(context_ids).issubset(set(retrieved_ids)):

            errors.append(

                f"Row {index}: selected context contains a document "

                "outside the retrieved BM25 candidate pool"

            )

        if row.get("retrieved_candidate_count") != len(retrieved_ids):

            errors.append(

                f"Row {index}: retrieved_candidate_count is inconsistent "

                "with retrieved_candidate_ids"

            )

        actual_context_chars = sum(

            len(str(passage.get("text", "")))

            for passage in passages

        )

        if actual_context_chars != row.get("context_chars"):

            errors.append(

                f"Row {index}: context_chars does not match passage text"

            )

        char_budget = row.get("context_char_budget")

        if not isinstance(char_budget, int) or isinstance(char_budget, bool):

            errors.append(f"Row {index}: invalid context_char_budget")

        elif actual_context_chars > char_budget:

            errors.append(

                f"Row {index}: context text exceeds its character budget"

            )

        # Validate Gemini's reported token counts.

        usage = row.get("usage")

        if not isinstance(usage, dict):

            errors.append(f"Row {index}: missing token usage metadata")

            if method in METHODS:

                token_totals[method]["missing_usage"] += 1

        else:

            token_fields = (

                "prompt_tokens",

                "completion_tokens",

                "total_tokens",

            )

            valid_tokens = all(

                isinstance(usage.get(field), int)

                and not isinstance(usage.get(field), bool)

                and usage[field] >= 0

                for field in token_fields

            )

            if not valid_tokens:

                errors.append(

                    f"Row {index}: token counts are missing or invalid"

                )

                if method in METHODS:

                    token_totals[method]["missing_usage"] += 1

            else:

                prompt_tokens = usage["prompt_tokens"]

                completion_tokens = usage["completion_tokens"]

                total_tokens = usage["total_tokens"]

                if (

                    total_tokens < prompt_tokens

                    or total_tokens < completion_tokens

                ):

                    errors.append(

                        f"Row {index}: total_tokens is inconsistent "

                        "with prompt/completion tokens"

                    )

                if method in METHODS:

                    totals = token_totals[method]

                    totals["prompt"] += prompt_tokens

                    totals["completion"] += completion_tokens

                    totals["total"] += total_tokens

                    totals["count"] += 1

        if method in METHODS and qid:

            by_query[qid][method] = row

    # Pairwise validation: all methods for a query must use the same pool,

    # claim, expected label, and model. Retrieval timing is shared by design.

    for qid, methods in by_query.items():

        missing_methods = METHODS - methods.keys()

        if missing_methods:

            errors.append(

                f"Query {qid}: missing methods {sorted(missing_methods)}"

            )

            continue

        expected_labels = {

            methods[method].get("expected_output")

            for method in METHODS

        }

        claim_texts = {

            methods[method].get("input")

            for method in METHODS

        }

        model_names = {

            methods[method].get("model")

            for method in METHODS

        }

        candidate_pools = {

            tuple(

                str(value)

                for value in methods[method].get(

                    "retrieved_candidate_ids", []

                )

            )

            for method in METHODS

        }

        retrieval_times = {

            methods[method].get("retrieval_latency_s")

            for method in METHODS

        }

        if len(expected_labels) != 1:

            errors.append(

                f"Query {qid}: expected labels differ across methods"

            )

        if len(claim_texts) != 1:

            errors.append(

                f"Query {qid}: claim texts differ across methods"

            )

        if len(model_names) != 1:

            errors.append(f"Query {qid}: Gemini models differ across methods")

        if len(candidate_pools) != 1:

            errors.append(

                f"Query {qid}: methods used different BM25 candidate pools"

            )

        if len(retrieval_times) != 1:
            warnings.append(
                f"Query {qid}: retrieval timings differ across method records; "
                "likely a resumed run. Candidate-pool equality is still checked."
            )

    # Display the timing values so they can be checked by eye.

    print(f"Result file: {result_path}")

    print(f"Model: {args.model}")

    print(f"Split: {args.split}")

    print(f"Record count: {len(records)}")

    print(f"Claims represented: {len(by_query)}")

    print()

    for qid in sorted(by_query, key=natural_key):

        print(f"Query {qid}:")

        for method in sorted(METHODS):

            row = by_query[qid].get(method)

            if row is None:

                print(f"  {method}: MISSING")

                continue

            print(

                f"  {method}: "

                f"ranking={row.get('scoring_latency_s')}s, "

                f"generation={row.get('generation_latency_s')}s, "

                f"end-to-end={row.get('end_to_end_latency_s')}s, "

                f"prediction={row.get('predicted_label')}, "

                f"expected={row.get('expected_output')}"

            )

    print("\nMean latency by method:")

    for method in sorted(METHODS):

        print(f"  {method}:")

        for field in TIMING_FIELDS:

            values = method_times[method][field]

            if values:

                print(f"    {field}: {mean(values):.4f}s")

    print("\nToken totals:")

    for method in sorted(METHODS):

        print(f"  {method}: {dict(token_totals[method])}")

    # Setup durations are not included in per-query end-to-end timing.

    if manifest_path.exists():

        try:

            manifest = json.loads(

                manifest_path.read_text(encoding="utf-8")

            )

        except (OSError, json.JSONDecodeError) as exc:

            errors.append(f"Could not read timing manifest: {exc}")

            manifest = {}

        print("\nOne-time setup timings (excluded from per-query latency):")

        for field in (

            "dataset_load_s",

            "bm25_index_build_s",

            "reranker_model_initialization_s",

            "jev_client_initialization_s",

            "gemini_client_initialization_s",

        ):

            print(f"  {field}: {manifest.get(field)}")

    else:

        print(f"\nWARNING: Timing manifest not found: {manifest_path}")

    if warnings:
        print(f"\nWARNINGS: {len(warnings)} issue(s)")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:

        print(f"\nVALIDATION FAILED: {len(errors)} issue(s)")

        for error in errors:

            print(f"  - {error}")

        return 1

    print("\nVALIDATION PASSED")

    print(

        "Stage timings, shared retrieval pools, contexts, labels, "

        "and token metadata passed validation."

    )

    return 0

if __name__ == "__main__":

    sys.exit(main())