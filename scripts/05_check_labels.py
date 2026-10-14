
import json
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from contextlab.data import load_scifact

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
TAR = Path("data/raw/scifact_original.tar.gz")
CANDIDATES = Path("data/candidates.jsonl")
OUT = Path("data/claim_labels.jsonl")
CLAIM_FILES = {"claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"}


def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_original_claims(tar_path):
    claims = {}

    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            if Path(member.name).name not in CLAIM_FILES:
                continue

            stream = archive.extractfile(member)
            if stream is None:
                continue

            for line in stream:
                record = json.loads(line)
                claims[norm(record["claim"])] = record

    return claims


def claim_label(evidence):
    labels = {
        item["label"]
        for annotations in evidence.values()
        for item in annotations
    }

    if not labels:
        return "NO_EVIDENCE"
    if labels == {"SUPPORT"}:
        return "SUPPORT"
    if labels == {"CONTRADICT"}:
        return "CONTRADICT"
    return "MIXED"


def main():
    if not TAR.exists():
        TAR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, TAR)

    _, queries, qrels = load_scifact()
    original = load_original_claims(TAR)

    ours = {}
    with CANDIDATES.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = row["query_id"]
            ours.setdefault(
                qid,
                {
                    "text": row["query_text"],
                    "split": row["split"],
                },
            )

    matched_files = Counter()
    labels = Counter()
    qrels_mismatches = []
    evidence_incomplete = []
    missing_claims = []
    output_rows = []

    for qid, query in ours.items():
        text = queries.get(qid, query["text"])
        record = original.get(norm(text))

        if record is None:
            missing_claims.append(qid)
            continue

        cited_ids = {str(x) for x in record.get("cited_doc_ids", [])}
        evidence = record.get("evidence", {})
        evidence_ids = {str(x) for x in evidence}
        qrel_ids = {str(x) for x in qrels.get(qid, set())}

        if qrel_ids != cited_ids:
            qrels_mismatches.append(
                {
                    "query_id": qid,
                    "qrels_only": sorted(qrel_ids - cited_ids),
                    "cited_only": sorted(cited_ids - qrel_ids),
                }
            )

        if evidence_ids != cited_ids:
            evidence_incomplete.append(
                {
                    "query_id": qid,
                    "cited_count": len(cited_ids),
                    "evidence_count": len(evidence_ids),
                    "cited_without_evidence": sorted(cited_ids - evidence_ids),
                }
            )

        label = claim_label(evidence)
        labels[label] += 1
        matched_files[record.get("id")] += 0  # Keep no dependency on file ordering.

        output_rows.append(
            {
                "query_id": qid,
                "split": query["split"],
                "claim": text,
                "label": label,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Queries: {len(ours)}")
    print(f"Matched original claims: {len(output_rows)}")
    print(f"Unmatched claims: {len(missing_claims)}")
    print(f"Qrels vs cited_doc_ids mismatches: {len(qrels_mismatches)}")
    print(f"Cited documents without evidence annotations: {len(evidence_incomplete)}")
    print(f"Claim labels: {dict(labels)}")

    if qrels_mismatches:
        print("First qrels mismatches:", qrels_mismatches[:10])
    if evidence_incomplete:
        print("First evidence differences:", evidence_incomplete[:10])
    if missing_claims:
        print("Unmatched query IDs:", missing_claims[:10])

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
