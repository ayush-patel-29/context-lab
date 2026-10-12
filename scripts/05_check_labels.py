import json
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
TAR = Path("data/raw/scifact_original.tar.gz")
CANDIDATES = Path("data/candidates.jsonl")
OUT = Path("data/claim_labels.jsonl")
CLAIM_FILES = ("claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl")


def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_original_claims(tar_path):
    """claim text (normalized) -> (file name, claim record)"""
    claims = {}
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            name = Path(member.name).name
            if name in CLAIM_FILES:
                for line in tf.extractfile(member):
                    rec = json.loads(line)
                    claims[norm(rec["claim"])] = (name, rec)
    return claims


def claim_label(evidence):
    labels = {e["label"] for evs in evidence.values() for e in evs}
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
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, TAR)

    original = load_original_claims(TAR)

    ours = {}  # query_id -> dict
    with CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = ours.setdefault(r["query_id"], {"text": r["query_text"], "split": r["split"],
                                                "n_relevant": r["n_relevant"], "gold_seen": set()})
            if r["label"]:
                q["gold_seen"].add(r["passage_id"])

    found_in, labels, doc_mismatch, rows = Counter(), Counter(), 0, []
    for qid, q in ours.items():
        hit = original.get(norm(q["text"]))
        if hit is None:
            labels["NOT_FOUND"] += 1
            continue
        fname, rec = hit
        evidence = rec.get("evidence", {})
        lab = claim_label(evidence)
        found_in[fname] += 1
        labels[lab] += 1
        # sanity: do the evidence documents agree with our relevance labels?
        if len(evidence) != q["n_relevant"] or not q["gold_seen"] <= set(evidence):
            doc_mismatch += 1
        rows.append({"query_id": qid, "split": q["split"], "claim": q["text"], "label": lab})

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"our queries: {len(ours)}   matched to an original claim: {len(rows)}")
    print("matched in file:", dict(found_in))
    print("claim labels:", dict(labels))
    print(f"evidence docs disagree with our relevance labels: {doc_mismatch}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()