# NovoMCP tool-search eval

A held-out benchmark for tool retrieval on an MCP server: given a user's
message, return the right tools from the catalog — and return nothing when
nothing applies.

The headline result is not the one that was expected.

| Retrieval mode | Recall@10 | Negatives passed |
|---|---|---|
| Keyword | 96.0% (24/25) | 80% (12/15) |
| Semantic | **100%** (25/25) | 47% (7/15) |
| Hybrid, unweighted RRF | **100%** (25/25) | 60% (9/15) |
| Hybrid, tuned on dev | **100%** (25/25) | 67% (10/15) |

Held-out test split, 39 cases. Cohere `embed-english-v3.0` at 1024 dimensions,
69-tool catalog, NovoMCP engine commit `36c968c`, Apple M3, 2026-10-27.

**Finding the right tool is solved. Declining to answer is not.** Every
semantic variant retrieves the correct tool for every query that has one.
More than half of the queries that should return nothing return something
anyway, and tuning moved that from 60% to 67%.

## Why the negative cases matter

An earlier version of this set had five negative cases, four of them easy
("what's the weather"). It scored 100% and reported a solved problem. This
version has thirty, including the two classes that actually break retrieval:

**On-topic questions that are not requests.** *"Explain how CYP450 enzymes
metabolize drugs"* retrieves `predict_admet` and `predict_bde`. *"Why do
proteins fold into specific shapes?"* retrieves three structure tools. The
embedding measures what a query is **about**; it carries no representation of
whether the user wants something run or something explained. Cosine similarity
cannot separate *explain docking* from *dock this*.

**Plausible asks the engine cannot do.** *"Design a CRISPR guide RNA"*,
*"optimize this peptide sequence"*, *"assemble this genome"*. Each is adjacent
to the catalog and outside it. Retrieval reaches for the nearest neighbour
rather than reporting that there isn't one.

Both classes point at a component this architecture does not have: an intent
gate in front of retrieval, deciding instruction versus question. That is a
classification a small model can do and an embedding cannot.

## The protocol

Written down before any result was seen, because a benchmark you can tune
against is a benchmark that measures the tuner.

- **Frozen set.** `eval-set.yaml`, 79 cases, sha256 `3097a7d2ea17222b3bfbad5a13c2abe170d9b97b85c6e045796d2c6e8514b6f0`.
  The 25 negatives added on 2026-10-27 were written against the tool catalog
  and plausible user intent, before any ranker was run on them.
- **Split.** Stratified by category; within a category, case ids are sorted and
  assigned alternately to dev and test. Deterministic, recomputed on every run,
  no hand-picking. 40 dev, 39 test.
- **Tuning.** On dev only. Grid: embedding weight `w` ∈ {1,2,3,4} against a
  keyword weight of 1; abstention threshold `T` on top-1 cosine similarity
  ∈ {0.00, 0.15, 0.20, 0.25, 0.30}.
- **Selection.** Maximise `(recall + negative_pass_rate) / 2` on dev.
- **Reporting.** The selected configuration runs **once** on test.

Two honest notes on this run. The embedding weight `w` had no effect at any
value, so fusion weighting is not the lever it was assumed to be. And the
selected threshold `T=0.30` sits at the edge of the grid, which means the grid
was too narrow — widening it belongs on dev, against a **fresh** test split.
Re-running this split after seeing its results would contaminate it.

## Scoring

- **Recall@10** — of the tools a case says must appear, how many are in the
  top ten. Aggregated over tools, not over cases.
- **Negative cases** — `expected: []` and a list of tools that must not appear
  in the top three. Scored separately; they are not in the recall denominator,
  which is exactly how a weak abstention rate hid behind a strong recall number.
- Whitelisted tools return regardless of query, so no whitelisted tool appears
  in `should_not_surface_in_top3`. Listing one would guarantee a failure that
  says nothing about retrieval.

## Running it

```bash
pip install -r requirements.txt

# Keyword mode — no credentials, no network
python run_eval_local.py eval-set.yaml

# Semantic + hybrid, with the full held-out protocol
export EMBEDDING_PROVIDER=cohere
export COHERE_API_KEY=...        # or OPENAI_API_KEY, or AZURE_OPENAI_API_KEY + ENDPOINT
python protocol.py
```

Both scripts import the engine in-process and call `tool_search.search()`
directly. The server's `POST /mcp/tool-search` route is a thin wrapper over the
same function, so the ranking measured is identical; what is skipped is the
network hop. `run_eval_http.py` is the original harness and hits a running
server instead — use it when latency is what you want to measure, not recall.

The scripts expect the engine source at `~/dev/novomcp/orchestrator/src`.

## What is not measured here

**Latency.** The in-process runners skip the network hop. A published query
time must come from the HTTP harness, against a stated provider from a stated
network position.

**Index size.** It tracks the embedding model's dimensions, not the catalog.
Cohere at 1024 dims gives 276 KB for 69 tools; a 1536-dimension model gives
roughly half as much again. Quoting an index size without its provider is
quoting the wrong variable.

**Any catalog but this one.** Results are specific to the 69 tools at engine
commit `36c968c`. A tool whose description changes is a different benchmark.

## Retired cases

Four cases were retired on 2026-10-27 because the tools they expect no longer
exist: `run_fep_check` went when the free-energy path left the open release,
and `predict_clinical_outcomes` was withdrawn. They measured the catalog rather
than the ranker, and no surviving tool is an honest substitute. The reason is
recorded in the eval set's header; the denominator moved from 58 cases to 54
before the new negatives were added.

## Provenance

Every number here carries the hardware it ran on, the model and dimensions, the
engine commit, the catalog size and the date. A figure without those cannot be
defended, and the point of the date is that it can later disagree with the
sentence around it.

Re-pull before citing.
