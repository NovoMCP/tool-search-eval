#!/usr/bin/env python3
"""Held-out evaluation of tool-search retrieval.

PROTOCOL, fixed before any result was seen:
  Split      stratified by category; within a category, case ids sorted and
             assigned alternately to dev/test. Deterministic, no hand-picking.
  Tuning     only on dev. Grid: embedding weight w in {1,2,3,4} against a
             keyword weight of 1; abstention threshold T on top-1 cosine
             similarity in {0.00,0.15,0.20,0.25,0.30}. Below T, retrieval
             returns nothing and only the whitelist stands.
  Selection  maximise (recall_rate + negative_pass_rate)/2 on dev.
  Reporting  the selected config is run ONCE on test. Baselines are run on
             the same test split for comparison.
Query embeddings are computed once and cached so that configs are compared
on identical vectors.
"""
import asyncio, os, sys, json, hashlib
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, os.environ.get("NOVOMCP_SRC", str(Path.home() / "dev/novomcp/orchestrator/src")))
import yaml, numpy as np
from novomcp.mcp import tool_search as ts

RRF_K, CAND, TOP_K = 60, 25, 10


def split(cases):
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)
    dev, test = [], []
    for cat in sorted(by_cat):
        for i, c in enumerate(sorted(by_cat[cat], key=lambda x: x["id"])):
            (dev if i % 2 == 0 else test).append(c)
    return dev, test


def rank(idx, qvec, query, w, T):
    """Fused ranking for one query under config (w, T)."""
    emb = ts._embedding_search(idx, qvec, CAND) if qvec is not None else []
    if T > 0 and (not emb or emb[0][1] < T):
        return []                      # abstain: whitelist only
    kw = ts._keyword_search(idx, query, CAND)
    s = defaultdict(float)
    for ranking, weight in ((emb, w), (kw, 1.0)):
        for r, (tool, _) in enumerate(ranking, 1):
            s[tool.name] += weight / (RRF_K + r)
    return sorted(s, key=lambda n: -s[n])[:TOP_K]


def with_whitelist(idx, names):
    out = list(names)
    for n in ts.CORE_WHITELIST:
        if n not in out and any(t.name == n for t in idx.tools):
            out.append(n)
    return out


def score(idx, cases, qvecs, w, T, mode="hybrid"):
    tot_e = tot_h = 0; neg_pass = neg_n = 0; fails = []; fps = []
    for c in cases:
        qv = qvecs[c["id"]]
        if mode == "semantic":
            names = [t.name for t, _ in ts._embedding_search(idx, qv, TOP_K)]
        elif mode == "keyword":
            names = [t.name for t, _ in ts._keyword_search(idx, c["query"], TOP_K)]
        else:
            names = rank(idx, qv, c["query"], w, T)
        names = with_whitelist(idx, names)
        exp = set(c.get("expected") or [])
        if exp:
            found = exp & set(names)
            tot_e += len(exp); tot_h += len(found)
            if exp - set(names): fails.append((c["id"], sorted(exp - set(names))))
        else:
            neg_n += 1
            fp = set(c.get("should_not_surface_in_top3") or []) & set(names[:3])
            if fp: fps.append((c["id"], sorted(fp), names[:3]))
            else: neg_pass += 1
    rec = tot_h / tot_e if tot_e else 0.0
    neg = neg_pass / neg_n if neg_n else 1.0
    return {"recall": rec, "neg": neg, "obj": (rec + neg) / 2,
            "hits": tot_h, "exp": tot_e, "neg_pass": neg_pass, "neg_n": neg_n,
            "fails": fails, "fps": fps}


async def main():
    data = yaml.safe_load(open("eval-set-v3.yaml"))
    cases = data["cases"]
    dev, test = split(cases)
    print(f"frozen sha256: {hashlib.sha256(open('eval-set-v3.yaml','rb').read()).hexdigest()[:16]}")
    print(f"cases {len(cases)}  dev {len(dev)}  test {len(test)}")

    await ts.build_index()
    idx = ts.get_index()
    print(f"index {idx.size} tools, fallback={idx.embedding_fallback}")

    qvecs = {}
    for c in cases:
        qvecs[c["id"]] = await ts._embed_query(c["query"])
    print(f"cached {sum(v is not None for v in qvecs.values())}/{len(cases)} query embeddings")

    print("\n── tuning on dev only ──")
    best, grid = None, []
    for w in (1, 2, 3, 4):
        for T in (0.00, 0.15, 0.20, 0.25, 0.30):
            r = score(idx, dev, qvecs, w, T)
            grid.append((w, T, r))
            if best is None or r["obj"] > best[2]["obj"]: best = (w, T, r)
    for w, T, r in grid:
        mark = " <-- selected" if (w, T) == (best[0], best[1]) else ""
        print(f"  w={w} T={T:.2f}  recall={r['recall']:.1%} neg={r['neg']:.0%} obj={r['obj']:.3f}{mark}")

    w, T, devr = best
    print(f"\nselected on dev: w={w}, T={T:.2f}")
    print("\n── TEST (held out, run once) ──")
    for name, mode in (("keyword", "keyword"), ("semantic", "semantic")):
        r = score(idx, test, qvecs, 1, 0.0, mode)
        print(f"  {name:22s} recall={r['recall']:.1%} ({r['hits']}/{r['exp']})  neg={r['neg']:.0%} ({r['neg_pass']}/{r['neg_n']})")
    r0 = score(idx, test, qvecs, 1, 0.0)
    print(f"  {'hybrid unweighted':22s} recall={r0['recall']:.1%} ({r0['hits']}/{r0['exp']})  neg={r0['neg']:.0%} ({r0['neg_pass']}/{r0['neg_n']})")
    rt = score(idx, test, qvecs, w, T)
    print(f"  {'hybrid tuned':22s} recall={rt['recall']:.1%} ({rt['hits']}/{rt['exp']})  neg={rt['neg']:.0%} ({rt['neg_pass']}/{rt['neg_n']})")
    if rt["fails"]:
        print("\n  recall misses:")
        for i, m in rt["fails"]: print(f"    {i}: {m}")
    if rt["fps"]:
        print("\n  negative false positives:")
        for i, f, t3 in rt["fps"]: print(f"    {i}: {f}")

asyncio.run(main())
