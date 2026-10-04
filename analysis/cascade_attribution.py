"""Root-cause cascade attribution for the Experiment-2 firm-level analysis.

Replaces the equal-split attribution of yan_nexus_test.py: instead of crediting
each seed with cascade/|S|, every cascade removal pushes one unit of "removal
mass" upstream along the recorded `disruption_origin` links (equal split across
a firm's valid origins, recursively) until the mass lands on seed firms. Seeds
with no downstream failures receive exactly 0. The attribution is
mass-conserving: per run, the seed masses sum to the cascade size (checked).

Removals with no traceable origin (empty/unknown `disruption_origin`) fall back
to a uniform split across the run's seeds; the fallback share is reported.

Outputs Pearson/Spearman/partial correlations of the per-firm mean attributed
mass against the five structural predictors of Table D1,
alongside the equal-split baseline computed from the same runs.

Output: outputs/figures/cascade_attribution_results.json
"""

import json
from collections import defaultdict

import networkx as nx
import numpy as np
import polars as pl
from networkx.algorithms.community import louvain_communities
from scipy import stats as scistats

from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

DATA_DIR = paths.DATA_DIR
RUNS_DIR = paths.RUNS_DIR / "exp2"

ANALYSIS_DIR = paths.ANALYSIS_DIR
# Directed node betweenness per firm (compute_node_betweenness.py). betweenness_centrality.json
# holds edge betweenness keyed by relation_id and must not be looked up by firm_id.
BETWEENNESS_PATH = ANALYSIS_DIR / "node_betweenness.json"

OUTPUT_PATH = paths.FIGURES_DIR / "cascade_attribution_results.json"


def attribute_run(rec: dict) -> tuple[dict[int, float], int, int, int]:
    """Return (seed -> attributed mass, cascade size, n_fallback, n_cycle_drops).

    Each cascade firm contributes 1 unit of mass, distributed over the seeds
    reachable via its `disruption_origin` chain (equal split per hop).
    """
    seeds = {s["id"] for s in rec.get("seeds", [])}
    removed = {r["id"]: r for r in rec["removed"]}
    cascade_ids = [i for i in removed if i not in seeds]
    mass = {s: 0.0 for s in seeds}
    if not cascade_ids:
        return mass, 0, 0, 0

    n_seeds = len(seeds)
    fallback = {s: 1.0 / n_seeds for s in seeds}
    n_fallback = 0
    n_cycle = 0
    memo: dict[int, dict[int, float]] = {}

    def dist(fid: int, guard: frozenset) -> dict[int, float] | None:
        """Seed distribution of firm fid, or None on a cycle."""
        nonlocal n_cycle
        if fid in seeds:
            return {fid: 1.0}
        if fid in memo:
            return memo[fid]
        if fid in guard:
            n_cycle += 1
            return None
        origins = removed[fid].get("disruption_origin") or []
        valid = [o for o in origins if o in seeds or o in removed]
        sub = []
        for o in valid:
            d = dist(o, guard | {fid})
            if d is not None:
                sub.append(d)
        if not sub:
            return None  # no traceable origin from here
        share = 1.0 / len(sub)
        out: dict[int, float] = defaultdict(float)
        for d in sub:
            for s, v in d.items():
                out[s] += v * share
        out = dict(out)
        memo[fid] = out
        return out

    for fid in cascade_ids:
        d = dist(fid, frozenset())
        if d is None:
            d = fallback
            n_fallback += 1
        for s, v in d.items():
            mass[s] += v

    return mass, len(cascade_ids), n_fallback, n_cycle


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    def residuals(a: np.ndarray) -> np.ndarray:
        slope, intercept, *_ = scistats.linregress(z, a)
        return a - (slope * z + intercept)

    r, p = scistats.pearsonr(residuals(x), residuals(y))
    return float(r), float(p)


def main() -> None:
    # --- Stream runs and accumulate per-firm attributions ---
    root_mass: dict[int, list[float]] = defaultdict(list)   # root-cause
    equal_mass: dict[int, list[float]] = defaultdict(list)  # equal-split baseline
    total_fallback = total_cascade = total_cycle = 0
    max_conservation_err = 0.0

    files = sorted(RUNS_DIR.glob("*.jsonl"))
    print(f"Streaming {len(files)} jsonl files ...")
    n_runs = 0
    for n, fp in enumerate(files, 1):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                mass, csize, nfb, ncyc = attribute_run(rec)
                total_fallback += nfb
                total_cascade += csize
                total_cycle += ncyc
                err = abs(sum(mass.values()) - csize)
                max_conservation_err = max(max_conservation_err, err)
                ns = len(mass)
                for s, v in mass.items():
                    root_mass[s].append(v)
                    equal_mass[s].append(csize / ns if ns else 0.0)
                n_runs += 1
        if n % 40 == 0:
            print(f"  {n}/{len(files)} files, {n_runs} runs")

    print(f"runs: {n_runs}, cascade removals: {total_cascade}, "
          f"fallback removals: {total_fallback} "
          f"({total_fallback / max(total_cascade, 1):.1%}), "
          f"cycle drops: {total_cycle}")
    print(f"max per-run mass-conservation error: {max_conservation_err:.2e}")

    # --- Structural predictors (same sources as yan_nexus_test.py) ---
    print("Loading MarkLines network ...")
    base_net, _, _ = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch
    )
    und = base_net.to_undirected()
    lcc_nodes = max(nx.connected_components(und), key=len)
    lcc = und.subgraph(lcc_nodes).copy()

    with open(BETWEENNESS_PATH, encoding="utf-8") as f:
        betweenness = {int(k): v for k, v in json.load(f).items()}
    print("Computing PageRank ...")
    pagerank = nx.pagerank(base_net, alpha=0.85)
    print("Computing Louvain (seed=42) + participation coefficient ...")
    communities = louvain_communities(lcc, seed=42)
    node_community = {n: i for i, com in enumerate(communities) for n in com}
    participation: dict[int, float] = {}
    for node in lcc.nodes():
        ki = lcc.degree(node)
        if ki == 0:
            participation[node] = 0.0
            continue
        c_count: dict[int, int] = defaultdict(int)
        for nb in lcc.neighbors(node):
            c_count[node_community[nb]] += 1
        participation[node] = 1.0 - sum((k / ki) ** 2 for k in c_count.values())

    rows = []
    for fid, masses in root_mass.items():
        if fid not in base_net.nodes:
            continue
        rows.append(
            {
                "firm_id": fid,
                "n_runs": len(masses),
                "mean_root_mass": float(np.mean(masses)),
                "mean_equal_mass": float(np.mean(equal_mass[fid])),
                "out_degree": int(base_net.out_degree(fid)),
                "in_degree": int(base_net.in_degree(fid)),
                "betweenness": float(betweenness.get(fid, 0.0)),
                "pagerank": float(pagerank.get(fid, 0.0)),
                "participation": float(participation.get(fid, 0.0)),
            }
        )
    df = pl.DataFrame(rows)
    print(f"firm-level dataset: {df.height} firms")

    predictors = ["out_degree", "in_degree", "betweenness", "pagerank", "participation"]
    results: dict = {
        "n_runs": n_runs,
        "n_firms": int(df.height),
        "total_cascade_removals": total_cascade,
        "fallback_removals": total_fallback,
        "fallback_share": total_fallback / max(total_cascade, 1),
        "cycle_drops": total_cycle,
        "max_conservation_error": max_conservation_err,
    }
    z = df["out_degree"].to_numpy().astype(float)
    for target in ["mean_root_mass", "mean_equal_mass"]:
        y = df[target].to_numpy()
        block: dict = {"pearson": {}, "spearman": {}, "partial_vs_out_degree": {}}
        print(f"\n=== target: {target} ===")
        for pred in predictors:
            x = df[pred].to_numpy().astype(float)
            r, p = scistats.pearsonr(x, y)
            rho, p_rho = scistats.spearmanr(x, y)
            block["pearson"][pred] = {"r": float(r), "p": float(p)}
            block["spearman"][pred] = {"rho": float(rho), "p": float(p_rho)}
            print(f"  {pred:>14}: r={r:+.3f} (p={p:.1e})  rho={rho:+.3f} (p={p_rho:.1e})")
        for pred in predictors:
            if pred == "out_degree":
                continue
            x = df[pred].to_numpy().astype(float)
            r, p = partial_corr(x, y, z)
            block["partial_vs_out_degree"][pred] = {"r": float(r), "p": float(p)}
            print(f"  {pred:>14} | out_degree: partial r={r:+.3f} (p={p:.1e})")
        results[target] = block

    # Top attributed firms for a sanity check against the critical-cascade events
    top = df.sort("mean_root_mass", descending=True).head(10)
    results["top_firms_by_root_mass"] = top.select(
        ["firm_id", "n_runs", "mean_root_mass", "mean_equal_mass", "out_degree"]
    ).to_dicts()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
