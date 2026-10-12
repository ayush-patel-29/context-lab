
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TAR = DATA / "raw" / "scifact_original.tar.gz"

LABELS_FILE = DATA / "claim_labels.jsonl"
CANDIDATES_FILE = DATA / "candidates.jsonl"

LABELED_FILES = {"claims_train.jsonl", "claims_dev.jsonl"}
UNLABELED_FILES = {"claims_test.jsonl"}
CLAIM_FILES = LABELED_FILES | UNLABELED_FILES

VALID_LABELS = {"SUPPORT", "CONTRADICT", "MIXED", "NO_EVIDENCE"}
VALID_SPLITS = {"dev", "test"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def label_from_evidence(evidence: dict) -> str:
    labels = {
        annotation["label"]
        for annotations in evidence.values()
        for annotation in annotations
        if isinstance(annotation, dict) and "label" in annotation
    }

    unexpected = labels - {"SUPPORT", "CONTRADICT"}
    if unexpected:
        raise ValueError(f"Unexpected SciFact evidence labels: {unexpected}")

    if not labels:
        return "NO_EVIDENCE"
    if labels == {"SUPPORT"}:
        return "SUPPORT"
    if labels == {"CONTRADICT"}:
        return "CONTRADICT"
    return "MIXED"


def load_original_data() -> tuple[dict, dict]:
    """Load original claims and corpus from the SciFact archive."""
    if not TAR.exists():
        raise FileNotFoundError(
            f"SciFact archive not found: {TAR}\n"
            "Run scripts/05_check_labels.py to download it first."
        )

    claims = {}
    corpus = {}

    with tarfile.open(TAR, "r:gz") as archive:
        for member in archive.getmembers():
            filename = Path(member.name).name

            if filename not in CLAIM_FILES and filename != "corpus.jsonl":
                continue

            stream = archive.extractfile(member)
            if stream is None:
                continue

            for line in stream:
                record = json.loads(line)

                if filename == "corpus.jsonl":
                    corpus[str(record["doc_id"])] = record
                    continue

                key = normalize(record["claim"])
                priority = 0 if filename in LABELED_FILES else 1

                previous = claims.get(key)

                # Prefer a labeled train/dev record over an unlabeled test record.
                if previous is None or priority < previous[0]:
                    claims[key] = (priority, filename, record)

    return {
        key: (filename, record)
        for key, (_, filename, record) in claims.items()
    }, corpus


def extract_reference_context(
    evidence: dict,
    corpus: dict,
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Extract the exact annotated sentences for each evidence rationale."""
    contexts = []
    evidence_doc_ids = []
    missing_doc_ids = []
    missing_sentence_groups = []

    for doc_id, annotations in evidence.items():
        doc_id = str(doc_id)
        evidence_doc_ids.append(doc_id)

        document = corpus.get(doc_id)

        if document is None:
            missing_doc_ids.append(doc_id)
            continue

        abstract = document.get("abstract", [])

        if not isinstance(annotations, list) or not annotations:
            missing_sentence_groups.append({
                "doc_id": doc_id,
                "reason": "missing_annotations",
            })
            continue

        for annotation in annotations:
            sentence_ids = annotation.get("sentences", [])

            if not isinstance(sentence_ids, list) or not sentence_ids:
                missing_sentence_groups.append({
                    "doc_id": doc_id,
                    "reason": "missing_sentence_indices",
                })
                continue

            selected_sentences = []
            invalid_ids = []

            for sentence_id in sentence_ids:
                if (
                    type(sentence_id) is int
                    and 0 <= sentence_id < len(abstract)
                    and isinstance(abstract[sentence_id], str)
                ):
                    selected_sentences.append(
                        abstract[sentence_id].strip()
                    )
                else:
                    invalid_ids.append(sentence_id)

            if selected_sentences:
                contexts.append(" ".join(selected_sentences))

            if invalid_ids:
                missing_sentence_groups.append({
                    "doc_id": doc_id,
                    "invalid_sentence_ids": invalid_ids,
                })

    # Deduplicate while preserving order.
    unique_contexts = []
    seen = set()

    for item in contexts:
        key = normalize(item)
        if key and key not in seen:
            seen.add(key)
            unique_contexts.append(item)

    return (
        unique_contexts,
        evidence_doc_ids,
        missing_doc_ids,
        missing_sentence_groups,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    labels = read_jsonl(LABELS_FILE)
    candidates = read_jsonl(CANDIDATES_FILE)
    original_claims, corpus = load_original_data()

    labels_by_id = {}

    for row in labels:
        qid = str(row["query_id"])

        if qid in labels_by_id:
            raise ValueError(f"Duplicate claim label: {qid}")

        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"Invalid split for {qid}: {row['split']}")

        if row["label"] not in VALID_LABELS:
            raise ValueError(f"Invalid label for {qid}: {row['label']}")

        labels_by_id[qid] = row

    candidates_by_id = defaultdict(list)

    for row in candidates:
        candidates_by_id[str(row["query_id"])].append(row)

    goldens = {"dev": [], "test": []}
    records = {"dev": [], "test": []}

    label_counts = Counter()
    source_file_counts = Counter()
    excluded_unlabeled = []
    missing_reference_context = []
    missing_reference_documents = {}
    missing_reference_sentences = {}

    bm25_zero_relevant = 0
    bm25_incomplete = 0

    for qid, row in labels_by_id.items():
        claim = row["claim"]
        split = row["split"]

        # Validate the candidate data for every query.
        query_rows = candidates_by_id.get(qid, [])

        if not query_rows:
            raise ValueError(f"No candidate passages found for {qid}")

        for candidate in query_rows:
            if candidate["split"] != split:
                raise ValueError(f"Split mismatch for {qid}")

            if normalize(candidate["query_text"]) != normalize(claim):
                raise ValueError(f"Claim text mismatch for {qid}")

        n_relevant_values = {
            int(candidate["n_relevant"]) for candidate in query_rows
        }

        if len(n_relevant_values) != 1:
            raise ValueError(f"Inconsistent n_relevant for {qid}")

        n_relevant = n_relevant_values.pop()

        retrieved_relevant_ids = {
            str(candidate["passage_id"])
            for candidate in query_rows
            if int(candidate["label"]) == 1
        }

        if n_relevant > 0 and not retrieved_relevant_ids:
            bm25_zero_relevant += 1

        if len(retrieved_relevant_ids) < n_relevant:
            bm25_incomplete += 1

        # Look up the authoritative SciFact annotation.
        source = original_claims.get(normalize(claim))

        if source is None:
            raise ValueError(f"Original SciFact claim not found: {qid}")

        source_file, original_record = source

        # The official test split has no usable gold labels.
        # Never convert its absent annotations into NO_EVIDENCE.
        if source_file in UNLABELED_FILES:
            excluded_unlabeled.append({
                "query_id": qid,
                "split": split,
                "source_file": source_file,
            })
            continue

        if "evidence" not in original_record:
            raise ValueError(
                f"Labeled source claim {qid} has no evidence field"
            )

        evidence = original_record["evidence"]

        if not isinstance(evidence, dict):
            raise ValueError(f"Invalid evidence structure for {qid}")

        authoritative_label = label_from_evidence(evidence)

        if row["label"] != authoritative_label:
            raise ValueError(
                f"Label mismatch for {qid}: "
                f"claim_labels.jsonl={row['label']}, "
                f"original SciFact={authoritative_label}. "
                "Regenerate claim_labels.jsonl before continuing."
            )

        (
            context,
            evidence_doc_ids,
            missing_doc_ids,
            missing_sentence_groups,
        ) = extract_reference_context(evidence, corpus)

        if authoritative_label != "NO_EVIDENCE" and not context:
            missing_reference_context.append(qid)

        if missing_doc_ids:
            missing_reference_documents[qid] = missing_doc_ids

        if missing_sentence_groups:
            missing_reference_sentences[qid] = missing_sentence_groups

        # Create a DeepEval Golden.
        metadata = {
            "query_id": qid,
            "split": split,
            "claim_label": authoritative_label,
            "source_claim_file": source_file,
            "evidence_doc_ids": evidence_doc_ids,
            "bm25_relevant_found": len(retrieved_relevant_ids),
            "bm25_relevant_total": n_relevant,
        }

        golden = Golden(
            input=claim,
            expected_output=authoritative_label,
            context=context,
            additional_metadata=metadata,
        )

        goldens[split].append(golden)

        records[split].append({
            "input": golden.input,
            "expected_output": golden.expected_output,
            "context": golden.context,
            "additional_metadata": metadata,
        })

        label_counts[authoritative_label] += 1
        source_file_counts[source_file] += 1

    # Validate and save the two separate DeepEval datasets.
    for split in ("dev", "test"):
        records[split].sort(
            key=lambda item: item["additional_metadata"]["query_id"]
        )

        goldens[split].sort(
            key=lambda item: item.additional_metadata["query_id"]
        )

        if goldens[split]:
            EvaluationDataset(goldens=goldens[split])

        output = DATA / f"goldens_{split}.jsonl"
        write_jsonl(output, records[split])

        print(f"{split}: {len(records[split])} goldens -> {output}")

    summary = {
        "total_goldens": sum(len(items) for items in records.values()),
        "by_split": {
            split: len(items) for split, items in records.items()
        },
        "by_label": dict(sorted(label_counts.items())),
        "source_claim_files": dict(sorted(source_file_counts.items())),
        "excluded_unlabeled_claim_count": len(excluded_unlabeled),
        "excluded_unlabeled_claims": excluded_unlabeled,
        "claims_missing_reference_context": missing_reference_context,
        "claims_with_missing_reference_documents": missing_reference_documents,
        "claims_with_missing_reference_sentences": missing_reference_sentences,
        "bm25_zero_relevant_candidates": bm25_zero_relevant,
        "bm25_incomplete_candidate_coverage": bm25_incomplete,
        "candidate_queries_without_claim_labels": len(
            set(candidates_by_id) - set(labels_by_id)
        ),
        "claim_labels_without_candidate_queries": len(
            set(labels_by_id) - set(candidates_by_id)
        ),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)

    (RESULTS / "golden_build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
