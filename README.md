# Context-Lab

**Investigating JEV for context engineering in LLM-powered systems.**

Context-Lab is an experimental research project exploring whether **JEV-based context operations can improve the quality, efficiency, and robustness of LLM-powered systems compared with conventional approaches.**

The central idea is that building better AI systems isn't only about choosing a more capable model. It's also about deciding what information the model receives, which tools it can use, what information it retains, and how context is managed throughout a task.

Rather than assuming a particular technique works, Context-Lab aims to evaluate these decisions through controlled experiments, measurable baselines, and reproducible results—including negative results.

## Research direction

The project investigates six dimensions of context engineering:

| Dimension | Research question |
|---|---|
| **Retrieval and RAG** | Can JEV select more useful evidence and improve downstream answers? |
| **Tool Loadout** | Can JEV select a smaller, more useful set of tools without reducing task success? |
| **Context Pruning** | Can JEV preserve more task-relevant information under a fixed context budget? |
| **Context Summarization** | Can JEV-guided selection preserve critical information in compressed context? |
| **Context Quarantine** | Can JEV help isolate untrusted or injected content without compromising legitimate tasks? |
| **Context Offloading** | Can JEV decide what information to store externally and retrieve when it is needed later? |

These are distinct research questions, not interchangeable techniques. Each requires its own baselines, evaluation criteria, and failure analysis.

The long-term objective is to identify **where JEV helps, where it does not, and whether any improvement justifies its computational and operational overhead.**

## Experiment 1 — Retrieval and RAG

**Status: Initial comparative evaluation completed; interpretation and reproducibility remain important.**

The first experiment investigates the relationship between evidence ranking, context selection, and downstream answer quality.

Three approaches were compared using the same BM25 top-30 candidate pool:

- **BM25:** lexical retrieval baseline.
- **Cross-encoder reranker:** conventional neural reranking baseline.
- **JEV:** an alternative approach to ranking candidate passages.

The downstream RAG evaluation used 280 held-out scientific claim-verification examples from a custom split of the available SciFact-derived dataset. The generator was Gemini 3.5 Flash Lite.

### Evaluation criteria

The experiment examined:

- Answer accuracy and macro-F1.
- Ranking and end-to-end latency.
- Prompt and total token usage.
- Document-level context precision and recall.
- Paired differences and 95% query-level bootstrap confidence intervals.

The comparison holds the candidate pool constant so the ranking strategies can be evaluated against the same retrieved documents.

### Results

The following are the observed results from the 280-example downstream RAG evaluation.

| Method | Answer accuracy | Mean ranking latency | Mean end-to-end latency |
|---|---:|---:|---:|
| BM25 | 78.57% | 0.05 s | 3.04 s |
| Cross-encoder reranker | **81.07%** | 3.48 s | 6.40 s |
| JEV | 78.21% | 0.99 s | 3.91 s |

*Latency excludes one-time index and model initialization. Ranking latency represents retrieval plus scoring; BM25 has no separate reranking stage.*

### What do these results suggest?

**1. The cross-encoder achieved the highest observed answer accuracy.**

Its accuracy was 81.07%, compared with 78.57% for BM25 and 78.21% for JEV.

However, the paired confidence intervals did not establish a conclusive improvement in answer quality over BM25. The observed differences should not be interpreted as proof of superiority.

**2. JEV ranked candidates faster than the cross-encoder.**

Mean ranking latency was approximately 0.99 seconds for JEV versus 3.48 seconds for the cross-encoder. JEV also had lower observed end-to-end latency than the cross-encoder in this configuration.

**3. Faster ranking did not translate into better observed answer accuracy.**

JEV's lower ranking overhead is an interesting efficiency result, but its answer accuracy was slightly below the BM25 baseline. This illustrates why ranking efficiency and downstream task quality must be evaluated separately.

**4. Better context relevance does not automatically prove better reasoning.**

Document-level context precision and recall measure relevance against the available document annotations. They do not, by themselves, establish semantic faithfulness or prove that a generated answer is supported by its cited evidence.

### Evaluation boundaries

- The 280 examples come from a custom held-out split, not necessarily the official SciFact test partition.
- All three ranking methods use the same top-30 BM25 candidate pool. The experiment does not evaluate documents that BM25 failed to retrieve.
- The paired confidence intervals do not establish a conclusive answer-quality improvement over BM25.
- Some resumed-run records have differing retrieval timings across methods. Candidate-pool consistency was validated, but these timing discrepancies should be considered when interpreting latency estimates.
- Token counts were recorded; monetary API cost was not calculated.

These limitations are part of the results, not details to hide.

## Previous retrieval-ranking results

A separate, previously recorded retrieval-only evaluation reported the following metrics on a 280-query split:

| Method | nDCG@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|
| BM25 | 0.652 | 0.619 | 0.772 |
| Cross-encoder reranker | 0.686 | 0.656 | 0.808 |
| JEV v1 | **0.696** | **0.670** | **0.819** |

These results suggest that JEV ranked relevant documents favorably in the recorded evaluation. However, the paired 95% confidence interval for the nDCG@10 difference between JEV and the cross-encoder included zero. These results are **previously recorded and have not been independently reproduced as part of the current protocol**.

Retrieval-only metrics and downstream RAG answer metrics measure different things and should not be treated as interchangeable evidence.

## Upcoming experiments

The next experiments will extend the investigation beyond retrieval. Each dimension will be evaluated independently before drawing broader conclusions about context engineering.

### 2. Context Pruning

**Question:** Can JEV preserve more useful information while reducing the context supplied to an LLM?

Potential baselines include truncation, recency-based selection, and conventional relevance ranking.

The evaluation will focus on downstream task success under equal context budgets, supported by evidence-retention, relevance, latency, and token measurements.

### 3. Context Summarization

**Question:** Can JEV-guided selection produce more useful compressed context than conventional summarization?

The comparison will use the same summarization and downstream models where feasible, with equal output budgets. Evaluation will consider task success, critical-fact retention, unsupported claims, compression ratio, and operational overhead.

### 4. Context Quarantine

**Question:** Can JEV help separate untrusted or adversarial content while preserving legitimate task utility?

This experiment will require a controlled suite of benign and adversarial inputs. Security outcomes will be evaluated alongside benign task success, false positives, missed attacks, and overhead.

### 5. Tool Loadout

**Question:** Can JEV identify which tools an agent needs for a task and avoid exposing unnecessary tools?

The comparison will consider all-tools-available, conventional tool retrieval, and other feasible selection baselines. The primary outcome will be task success, with tool-selection quality and execution overhead as secondary measures.

### 6. Context Offloading

**Question:** Can JEV improve decisions about what information to store outside the active context and what to retrieve later?

This experiment will use tasks that require information across multiple steps. Evaluation will consider delayed-fact recall, long-horizon task success, stale or incorrect retrievals, storage requirements, and latency.

*These are planned research directions, not completed experiments or established results.*

## Experimental principles

Context-Lab follows a common set of principles across its experiments:

1. **Use credible baselines.** Compare JEV with conventional methods appropriate to each research question.
2. **Control the comparison.** Keep task instances, available information, underlying models, prompts, and budgets fixed wherever possible.
3. **Separate development and held-out evaluation.** Use development data for tuning and freeze configurations before evaluating held-out data.
4. **Measure uncertainty.** Report sample sizes, paired differences, confidence intervals, and meaningful failure cases—not just point estimates.
5. **Measure quality and cost together.** Track task outcomes alongside tokens, latency, calls, and cost where available.
6. **Reuse cached artifacts.** Avoid unnecessary data downloads, recomputation, and paid model requests.
7. **Report negative results.** A technique that fails to improve the baseline is still a useful experimental finding when the evaluation is sound.

No single aggregate score is intended to collapse retrieval quality, task success, security, memory retention, and efficiency into one misleading measure. Comparisons should first be made within each research dimension.

## Repository and reproducibility

The repository contains experimental code, datasets or derived artifacts, evaluation scripts, and result reports as they are developed.

The intended workflow for each experiment is:

1. Define the research question and hypothesis.
2. Specify the dataset, baselines, primary metric, budgets, and stopping criteria.
3. Validate cached data and instrumentation before expensive runs.
4. Run a small pilot to verify the evaluation pipeline.
5. Freeze the experimental configuration before held-out evaluation.
6. Record results, uncertainty, latency, token usage, failures, and limitations.
7. Publish the interpretation with enough detail to distinguish measured findings from hypotheses.

Reproducibility depends on the exact dataset versions, model and prompt configurations, dependencies, cached artifacts, and execution commands. These details should be documented alongside each experiment rather than inferred from headline metrics.

## Project status

| Area | Status |
|---|---|
| RAG answer-quality comparison | Initial 280-example evaluation completed |
| Retrieval-only comparison | Previously recorded results; independent reproduction pending |
| Context pruning | Next candidate for development |
| Context summarization | Planned |
| Context quarantine | Planned |
| Tool loadout | Planned |
| Context offloading | Planned |

The sequence may change as data availability, feasibility, and pilot results become clearer.

## Research objective

Context engineering is often discussed as a collection of useful techniques. Context-Lab aims to investigate a more specific question:

**Which context operations measurably improve an AI system, under which conditions, and at what cost?**

The goal is not to demonstrate that JEV wins every comparison. It is to produce careful, reproducible evidence about where it offers an advantage—and where conventional methods remain the better choice.

---

*Context-Lab is an ongoing experimental project. Results are specific to the reported datasets, methods, and configurations; they should not be generalized beyond the available evidence.*