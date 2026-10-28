#!/usr/bin/env python3
"""In-process runner for the tool-search eval.

Calls novomcp.mcp.tool_search.search() directly instead of POSTing to a
running server. The HTTP route at /mcp/tool-search is a thin wrapper over
this same function, so the ranking path measured is identical; what is
skipped is the network hop, which is why this reports recall and not latency.
"""
import asyncio, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ.get("NOVOMCP_SRC", str(Path.home() / "dev/novomcp/orchestrator/src")))
import yaml
from novomcp.mcp import tool_search as ts


async def main(path):
    data = yaml.safe_load(open(path))
    cases = data["cases"]
    top_k = int(data.get("top_k", 10))

    await ts.build_index()
    st = ts.status()
    mode = "keyword (fallback)" if st.get("embedding_fallback") else "semantic"
    print(f"index: {st['size']} tools, mode={mode}, "
          f"build={st.get('build_duration_seconds')}s, ready={st['ready']}")
    if st.get("last_error"):
        print(f"last_error: {st['last_error']}")
    print("─" * 72)

    by_cat = defaultdict(list)
    total_expected = total_hit = 0
    failures, misfires = [], []

    for c in cases:
        res = await ts.search(c["query"], top_k=top_k)
        returned = [t["name"] for t in res.get("tools", [])]
        expected = set(c.get("expected") or [])
        found, missing = expected & set(returned), expected - set(returned)
        should_not = set(c.get("should_not_surface_in_top3") or [])
        fp = should_not & set(returned[:3])
        cat = c.get("category", "uncategorized")

        if expected:
            by_cat[cat].append(len(found) / len(expected))
            total_expected += len(expected); total_hit += len(found)
            ok = not missing
            if not ok:
                failures.append((c["id"], sorted(missing), returned[:5]))
            print(f"  {'PASS' if ok else 'FAIL'}  {c['id']:42s} "
                  f"found={len(found)}/{len(expected)}")
        else:
            ok = not fp
            by_cat[cat].append(1.0 if ok else 0.0)
            if not ok:
                misfires.append((c["id"], sorted(fp)))
            print(f"  {'PASS' if ok else 'FAIL'}  {c['id']:42s} top3={returned[:3]}")

    overall = total_hit / total_expected if total_expected else 0.0
    print("─" * 72)
    print(f"Overall recall@10: {overall:.1%}  ({total_hit}/{total_expected} tools found)")
    print(f"Cases: {len(cases)}   gate: {data.get('ship_gate_recall_at_10')}")
    print("\nBy category:")
    for cat, r in sorted(by_cat.items()):
        print(f"  {cat:16s} {sum(r)/len(r):6.1%}  (n={len(r)})")
    if failures:
        print("\nFailures:")
        for cid, miss, top5 in failures:
            print(f"  {cid}\n     missing: {miss}\n     top5:    {top5}")
    if misfires:
        print("\nNegative-case false positives:")
        for cid, fp in misfires:
            print(f"  {cid}: {fp}")
    return overall


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "eval-set.yaml"
    gate = float(yaml.safe_load(open(p)).get("ship_gate_recall_at_10") or 0.0)
    recall = asyncio.run(main(p))
    # The gate is recall only. Abstention is reported above and deliberately
    # not gated: keyword mode passes most negatives by coincidence, and a
    # threshold on a coincidence is not a regression test.
    if recall < gate:
        print(f"\nGATE FAIL: recall@10 {recall:.1%} is below the declared {gate:.0%}")
        sys.exit(1)
    print(f"\nGATE PASS: recall@10 {recall:.1%} at or above the declared {gate:.0%}")
