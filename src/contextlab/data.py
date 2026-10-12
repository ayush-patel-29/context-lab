import random

DATASET = "beir/scifact/test"


def load_scifact():
    """Return (docs, queries, qrels) as plain dicts.

    docs:    doc_id   -> "title. text"
    queries: query_id -> query text
    qrels:   query_id -> set of relevant doc_ids
    """
    import ir_datasets  # imported here so tests don't need the dataset

    ds = ir_datasets.load(DATASET)

    docs = {}
    for d in ds.docs_iter():
        title = getattr(d, "title", "") or ""
        docs[d.doc_id] = f"{title}. {d.text}" if title else d.text

    queries = {q.query_id: q.text for q in ds.queries_iter()}

    qrels = {}
    for r in ds.qrels_iter():
        if r.relevance > 0:
            qrels.setdefault(r.query_id, set()).add(r.doc_id)

    return docs, queries, qrels


def make_split(query_ids, n_total=100, n_dev=20, seed=42):
    """Pick n_total queries, return (dev_ids, test_ids). Deterministic."""
    ids = sorted(set(query_ids))  # sort first: set order is not stable
    if n_total > len(ids):
        raise ValueError(f"asked for {n_total} queries, only {len(ids)} available")
    if not 0 < n_dev < n_total:
        raise ValueError("n_dev must be between 1 and n_total - 1")

    chosen = random.Random(seed).sample(ids, n_total)
    return chosen[:n_dev], chosen[n_dev:]