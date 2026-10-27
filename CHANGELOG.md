# Changelog

## 2026-10-27 — first open baseline

- Eval set frozen at 79 cases, sha256 `3097a7d2…`.
- 25 negative cases added, written blind against the catalog before any
  ranker was run on them. Negatives go from 5 to 30.
- Four cases retired: `run_fep_check` and `predict_clinical_outcomes` are no
  longer in the catalog.
- Held-out protocol introduced: stratified deterministic split, tuning on dev
  only, test run once.
- Baseline recorded in `results/2026-10-27-baseline.json`.
- Finding: recall is solved (100% on held-out cases for every semantic
  variant); abstention is not (67% at best).

### Corrections to earlier published figures

The April 2026 figure of 100% recall was measured on a 62-tool catalog. A
later commit changed the tool count in the prose to 68 without re-measuring,
so the number came to sit beside a catalog it was never run against. It also
described the set as 50 prompts across 6 categories; it was 58 across 7.
