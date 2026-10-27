#!/usr/bin/env python3
"""Run the tool-search eval set against the live /mcp/tool-search endpoint.

Usage:
    export NOVOMCP_API_KEY=sk-...
    python3 scripts/tool-discovery/run_eval.py                    # default: production
    python3 scripts/tool-discovery/run_eval.py --base $STAGING    # staging
    python3 scripts/tool-discovery/run_eval.py --require-recall 0.90   # CI gate

Exits 0 if recall@10 ≥ --require-recall, non-zero otherwise. By default
the gate is off (just prints the score) so you can run ad-hoc without
failing.

Reports:
    - Overall recall@10
    - Per-category recall@10 (funnel / compute / materials / ambiguous /
      adversarial / negative)
    - Failed cases with expected-but-missing tool names
    - Negative-case false positives (tools that appeared in top-3 when
      they shouldn't have)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Missing dependency: httpx. Install with `pip install httpx pyyaml`.",
          file=sys.stderr)
    sys.exit(2)

try:
    import yaml
except ImportError:
    print("Missing dependency: pyyaml. Install with `pip install httpx pyyaml`.",
          file=sys.stderr)
    sys.exit(2)


DEFAULT_EVAL_PATH = Path(__file__).parent / "eval-set.yaml"
DEFAULT_BASE_URL = "https://api.novomcp.com"


def load_eval(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run_case(client: httpx.Client, case: dict, top_k: int) -> dict:
    """Query the endpoint for one case and compute hit/miss."""
    payload = {"query": case["query"], "top_k": top_k}
    try:
        r = client.post("/mcp/tool-search", json=payload)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {e}", "returned": [], "top3": []}

    returned = [t["name"] for t in data.get("tools", [])]
    top3 = returned[:3]

    expected = set(case.get("expected") or [])
    found = expected & set(returned)
    missing = expected - set(returned)

    should_not = set(case.get("should_not_surface_in_top3") or [])
    false_positives = should_not & set(top3)

    return {
        "returned": returned,
        "top3": top3,
        "expected": sorted(expected),
        "found": sorted(found),
        "missing": sorted(missing),
        "false_positives": sorted(false_positives),
        "recall": len(found) / len(expected) if expected else 1.0,
    }


def format_report(cases: list, results: list) -> float:
    """Print the per-case breakdown + aggregates. Returns overall recall@10."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    total_expected = 0
    total_hit = 0
    failures = []
    negative_misfires = []

    for case, result in zip(cases, results):
        if "error" in result:
            print(f"  ERROR  {case['id']:40s} — {result['error']}")
            continue

        cat = case.get("category", "uncategorized")
        expected = result["expected"]
        if expected:
            by_cat[cat].append(result["recall"])
            total_expected += len(expected)
            total_hit += len(result["found"])

            if result["missing"]:
                status = "FAIL"
                failures.append((case, result))
            else:
                status = "PASS"
            print(
                f"  {status}  {case['id']:40s} "
                f"recall={result['recall']:.0%}  "
                f"found={len(result['found'])}/{len(expected)}"
            )
        else:
            # Negative cases — pass iff no false positives in top-3
            if result["false_positives"]:
                status = "FAIL"
                negative_misfires.append((case, result))
            else:
                status = "PASS"
            by_cat[cat].append(1.0 if not result["false_positives"] else 0.0)
            print(
                f"  {status}  {case['id']:40s} "
                f"top3={result['top3']}"
            )

    overall = total_hit / total_expected if total_expected else 0.0

    print("\n" + "─" * 72)
    print(f"Overall recall@10: {overall:.1%}  ({total_hit}/{total_expected} tools found)")
    print("\nBy category:")
    for cat, recalls in sorted(by_cat.items()):
        avg = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"  {cat:15s} {avg:.1%}   ({len(recalls)} cases)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for case, result in failures:
            print(f"  {case['id']}")
            print(f"    query:    {case['query']}")
            print(f"    expected: {result['expected']}")
            print(f"    missing:  {result['missing']}")
            print(f"    returned: {result['returned'][:10]}")

    if negative_misfires:
        print(f"\nNegative misfires ({len(negative_misfires)}):")
        for case, result in negative_misfires:
            print(f"  {case['id']}")
            print(f"    query:            {case['query']}")
            print(f"    false positives:  {result['false_positives']}")
            print(f"    top-3:            {result['top3']}")

    return overall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("NOVOMCP_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key", default=os.environ.get("NOVOMCP_API_KEY") or os.environ.get("API"))
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    ap.add_argument("--require-recall", type=float, default=0.0,
                    help="Exit non-zero if overall recall@10 falls below this. "
                         "Default 0 (report only, don't gate).")
    ap.add_argument("--json", action="store_true",
                    help="Emit raw results as JSON to stdout instead of the text report.")
    args = ap.parse_args()

    if not args.api_key:
        print("Missing API key. Set NOVOMCP_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    eval_data = load_eval(args.eval)
    cases = eval_data["cases"]
    top_k = int(eval_data.get("top_k", 10))

    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(base_url=args.base, headers=headers, timeout=30.0) as client:
        # Quick status check so we know the index is ready before we run 50 queries.
        try:
            s = client.get("/mcp/tool-search/status").json()
            print(f"Index: ready={s.get('ready')} size={s.get('size')} "
                  f"dims={s.get('embedding_dimensions')} "
                  f"fallback={s.get('embedding_fallback')} "
                  f"built_in={s.get('build_duration_seconds')}s\n")
            if not s.get("ready"):
                print("WARNING: tool-search index not ready. Results may be empty "
                      "or degraded. Retry in ~5s if this is just after deploy.\n",
                      file=sys.stderr)
        except httpx.HTTPError as e:
            print(f"WARNING: could not read /mcp/tool-search/status: {e}\n",
                  file=sys.stderr)

        results = [run_case(client, case, top_k) for case in cases]

    if args.json:
        print(json.dumps({
            "cases": cases,
            "results": results,
        }, indent=2))
        return 0

    overall = format_report(cases, results)

    if args.require_recall > 0 and overall < args.require_recall:
        print(f"\nFAIL: recall@10 {overall:.1%} below gate {args.require_recall:.1%}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
