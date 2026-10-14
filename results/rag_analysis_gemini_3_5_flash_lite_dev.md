# RAG experiment analysis

- Split: `dev`
- Model: `gemini-3.5-flash-lite`
- Claims: **20**
- Bootstrap samples: 2000
- Reference labels represented: CONTRADICT, MIXED, NO_EVIDENCE, SUPPORT

Intervals are 95% query-level percentile intervals. Method differences versus BM25 are paired on the same resampled claims.

## Answer quality, latency, and context

| Method | Accuracy (95% CI) | Macro-F1 (95% CI) | Δ Accuracy vs BM25 (95% CI) | Δ Macro-F1 vs BM25 (95% CI) | Retrieval + scoring (s) | Scoring only (s) | Gemini generation (s) | End-to-end (s) | Mean prompt tokens | Mean total tokens | Context precision | Context recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.650 [0.450, 0.850] | 0.434 [0.269, 0.590] | — | — | 0.06 | 0.00 | 3.56 | 3.62 | 2023.7 | 2026.5 | 0.170 | 0.775 |
| reranker | 0.600 [0.400, 0.800] | 0.439 [0.250, 0.600] | -0.050 [-0.250, 0.100] | 0.005 [-0.127, 0.148] | 3.66 | 3.60 | 3.12 | 6.78 | 2127.6 | 2130.7 | 0.160 | 0.725 |
| jev_v1_n30 | 0.650 [0.450, 0.850] | 0.434 [0.273, 0.588] | 0.000 [-0.150, 0.150] | 0.000 [-0.076, 0.079] | 0.90 | 0.84 | 2.95 | 3.85 | 2028.4 | 2031.2 | 0.147 | 0.625 |

## Per-class metrics

| Method | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| bm25 | CONTRADICT | 3 | 0.250 | 0.333 | 0.286 |
| bm25 | MIXED | 0 | 0.000 | 0.000 | 0.000 |
| bm25 | NO_EVIDENCE | 8 | 0.625 | 0.625 | 0.625 |
| bm25 | SUPPORT | 9 | 0.875 | 0.778 | 0.824 |
| reranker | CONTRADICT | 3 | 0.400 | 0.667 | 0.500 |
| reranker | MIXED | 0 | 0.000 | 0.000 | 0.000 |
| reranker | NO_EVIDENCE | 8 | 0.556 | 0.625 | 0.588 |
| reranker | SUPPORT | 9 | 0.833 | 0.556 | 0.667 |
| jev_v1_n30 | CONTRADICT | 3 | 0.250 | 0.333 | 0.286 |
| jev_v1_n30 | MIXED | 0 | 0.000 | 0.000 | 0.000 |
| jev_v1_n30 | NO_EVIDENCE | 8 | 0.625 | 0.625 | 0.625 |
| jev_v1_n30 | SUPPORT | 9 | 0.875 | 0.778 | 0.824 |

## Resource totals

| Method | Prompt tokens | Completion tokens | Total tokens | Mean context chars | Mean passages | Invalid predictions | Missing token records |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 40474 | 56 | 40530 | 7956.5 | 4.80 | 0 | 0 |
| reranker | 42551 | 62 | 42613 | 8288.1 | 4.80 | 0 | 0 |
| jev_v1_n30 | 40568 | 56 | 40624 | 7932.9 | 4.65 | 0 | 0 |

## Per-query predictions

| Query | Expected | BM25 | Reranker | JEV |
|---|---|---|---|---|
| 124 | SUPPORT | SUPPORT | SUPPORT | SUPPORT |
| 142 | CONTRADICT | CONTRADICT | CONTRADICT | CONTRADICT |
| 198 | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE |
| 213 | NO_EVIDENCE | CONTRADICT | CONTRADICT | CONTRADICT |
| 230 | SUPPORT | SUPPORT | SUPPORT | SUPPORT |
| 239 | SUPPORT | NO_EVIDENCE | NO_EVIDENCE | SUPPORT |
| 312 | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE |
| 619 | CONTRADICT | NO_EVIDENCE | CONTRADICT | NO_EVIDENCE |
| 628 | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE |
| 808 | SUPPORT | CONTRADICT | CONTRADICT | CONTRADICT |
| 887 | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE |
| 960 | SUPPORT | SUPPORT | NO_EVIDENCE | NO_EVIDENCE |
| 1086 | SUPPORT | SUPPORT | SUPPORT | SUPPORT |
| 1088 | CONTRADICT | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE |
| 1099 | NO_EVIDENCE | SUPPORT | SUPPORT | SUPPORT |
| 1100 | NO_EVIDENCE | CONTRADICT | CONTRADICT | CONTRADICT |
| 1204 | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE | NO_EVIDENCE |
| 1216 | SUPPORT | SUPPORT | SUPPORT | SUPPORT |
| 1266 | SUPPORT | SUPPORT | NO_EVIDENCE | SUPPORT |
| 1290 | SUPPORT | SUPPORT | SUPPORT | SUPPORT |

## Interpretation cautions

- A small development set is not sufficient to establish method superiority.
- BM25 has no separate reranking stage; its search time is included in retrieval + scoring.
- End-to-end latency excludes one-time index and model initialization.
- Generation latency can fluctuate independently of the ranking method.
- Context precision/recall measure document relevance, not semantic faithfulness.
- This report records token counts, not API cost.

Machine-readable report: `rag_analysis_gemini_3_5_flash_lite_dev.json`
