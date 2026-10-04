"""Yan-nexus test: extend the critical-node regression with betweenness and
Louvain participation coefficient, and report partial correlations controlling
for out-degree.

The output is a JSON with Pearson and partial-r values (Table D1, run level).
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
RESULTS_DIR = paths.CONFIGS_DIR
MC_DIR = paths.RUNS_DIR / "exp2"
CONFIGS_PATH = RESULTS_DIR / "0_monte_carlo_configs_rev.json"

ANALYSIS_DIR = paths.ANALYSIS_DIR
# Directed node betweenness per firm (compute_node_betweenness.py). betweenness_centrality.json
# holds edge betweenness keyed by relation_id and must not be looked up by firm_id.
BETWEENNESS_PATH = ANALYSIS_DIR / "node_betweenness.json"

OUTPUT_PATH = paths.FIGURES_DIR / "yan_nexus_results.json"


def participation_coefficient(
    g: nx.Graph, partition: dict[int, int]
) -> dict[int, float]:
    """Guimera-Amaral participation coefficient.

    P_i = 1 - sum_c (k_ic / k_i)^2 where k_ic is the count of node i's neighbours
    in community c. P_i = 0 means all neighbours sit in one community; P_i -> 1
    means neighbours are spread across many communities (bridge node).
    """
    pc: dict[int, float] = {}
    for n in g.nodes():
        ki = g.degree(n)
        if ki == 0:
            pc[n] = 0.0
            continue
        c_count: dict[int, int] = {}
        for nb in g.neighbors(n):
            c = partition[nb]
            c_count[c] = c_count.get(c, 0) + 1
        pc[n] = 1.0 - sum((k / ki) ** 2 for k in c_count.values())
    return pc


def partial_corr(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Partial correlation of x and y controlling for z, with two-sided p-value."""
    # Residualise x and y against z via OLS
    def residuals(a: np.ndarray) -> np.ndarray:
        slope, intercept, *_ = scistats.linregress(z, a)
        return a - (slope * z + intercept)

    r, p = scistats.pearsonr(residuals(x), residuals(y))
    return float(r), float(p)


def main() -> None:
    print("Loading MarkLines network ...")
    base_net, _, _ = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch
    )
    print(f"  base_net: {base_net.number_of_nodes()} nodes, "
          f"{base_net.number_of_edges()} edges")

    # Largest weakly connected component (undirected) for Louvain + LCC analyses
    und = base_net.to_undirected()
    lcc_nodes = max(nx.connected_components(und), key=len)
    lcc = und.subgraph(lcc_nodes).copy()
    print(f"  LCC: {lcc.number_of_nodes()} nodes, {lcc.number_of_edges()} edges")

    # --- Centralities ---
    print("Loading betweenness centrality from disk ...")
    with open(BETWEENNESS_PATH, encoding="utf-8") as f:
        betweenness_str = json.load(f)
    # Keys are strings in JSON; convert to int
    betweenness = {int(k): v for k, v in betweenness_str.items()}
    print(f"  betweenness: {len(betweenness)} entries")

    print("Computing PageRank on directed network ...")
    pagerank = nx.pagerank(base_net, alpha=0.85)
    print(f"  pagerank: {len(pagerank)} entries")

    # --- Louvain communities + participation coefficient ---
    print("Computing Louvain communities on LCC (seed=42) ...")
    communities = louvain_communities(lcc, seed=42)
    print(f"  {len(communities)} communities")
    node_community = {n: i for i, com in enumerate(communities) for n in com}

    print("Computing participation coefficient on LCC ...")
    participation = participation_coefficient(lcc, node_community)

    # --- Build per-seed-firm regression dataset ---
    print("Loading Experiment-2 configs and stats ...")
    with open(CONFIGS_PATH, encoding="utf-8") as f:
        configs = json.load(f)
    stats = []
    for path in sorted(MC_DIR.glob("stats_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                stats.append({"task_id": r["task_id"], "num_removed": r["num_removed"][-1]})
    print(f"  configs: {len(configs)}, stats: {len(stats)}")

    # task_id -> (seed_firm_ids, num_seeds)
    task_to_info: dict[int, tuple[list[int], int]] = {}
    for c in configs:
        sc = c["simulation_config"]
        task_to_info[c["task_id"]] = (
            list(sc["disruption_seeds"]),
            int(sc["num_disruption_seeds"]),
        )

    # Per-seed-firm cascade contribution = (num_removed - num_seeds) / num_seeds
    # attributed equally to each seed in the run, then averaged over runs the
    # firm appears in as a seed.
    firm_contribs: dict[int, list[float]] = defaultdict(list)
    run_rows: list[dict] = []
    for r in stats:
        seeds, ns = task_to_info[r["task_id"]]
        nr = r["num_removed"]
        if isinstance(nr, list):
            nr = nr[-1]
        cascade = nr - ns
        per_seed = cascade / ns if ns > 0 else 0.0
        for s in seeds:
            firm_contribs[s].append(per_seed)

        # Run-level aggregates: mean of each predictor over the seed set
        valid_seeds = [s for s in seeds if s in base_net.nodes]
        if not valid_seeds:
            continue
        mean_pred = {}
        for name, getter in (
            ("out_degree", lambda f: base_net.out_degree(f)),
            ("in_degree", lambda f: base_net.in_degree(f)),
            ("betweenness", lambda f: betweenness.get(f, 0.0)),
            ("pagerank", lambda f: pagerank.get(f, 0.0)),
            ("participation", lambda f: participation.get(f, 0.0)),
        ):
            vals = [getter(s) for s in valid_seeds]
            mean_pred[f"mean_{name}"] = float(np.mean(vals))
            mean_pred[f"sum_{name}"] = float(np.sum(vals))
        run_rows.append(
            {
                "task_id": r["task_id"],
                "num_seeds": ns,
                "num_removed": nr,
                "cascade_size": cascade,
                **mean_pred,
            }
        )

    print(f"  unique seed firms with cascade observations: {len(firm_contribs)}")
    print(f"  run-level rows with valid seeds: {len(run_rows)}")

    # Build regression rows
    rows = []
    skipped = 0
    for f, contribs in firm_contribs.items():
        if f not in base_net.nodes:
            skipped += 1
            continue
        rows.append(
            {
                "firm_id": f,
                "n_runs": len(contribs),
                "mean_cascade_contrib": float(np.mean(contribs)),
                "median_cascade_contrib": float(np.median(contribs)),
                "out_degree": int(base_net.out_degree(f)),
                "in_degree": int(base_net.in_degree(f)),
                "w_out_degree": int(base_net.out_degree(f, weight="weight")),
                "w_in_degree": int(base_net.in_degree(f, weight="weight")),
                "betweenness": float(betweenness.get(f, 0.0)),
                "pagerank": float(pagerank.get(f, 0.0)),
                "participation": float(participation.get(f, 0.0)),
                "in_lcc": int(f in lcc_nodes),
            }
        )
    if skipped:
        print(f"  skipped {skipped} seed firms not in base_net")

    df = pl.DataFrame(rows)
    print(f"  regression dataset: {df.height} rows")

    # --- Pearson r for each predictor against mean cascade contribution ---
    predictors = [
        "out_degree",
        "w_out_degree",
        "in_degree",
        "w_in_degree",
        "betweenness",
        "pagerank",
        "participation",
    ]
    y = df["mean_cascade_contrib"].to_numpy()
    results: dict[str, dict[str, float | int]] = {
        "n_firms": int(df.height),
        "n_lcc_firms": int(df.filter(pl.col("in_lcc") == 1).height),
        "pearson": {},
        "spearman": {},
        "partial_vs_out_degree": {},
    }
    print("\n=== Pearson / Spearman vs mean cascade contribution per firm ===")
    for pred in predictors:
        x = df[pred].to_numpy().astype(float)
        r, p = scistats.pearsonr(x, y)
        rho, p_rho = scistats.spearmanr(x, y)
        results["pearson"][pred] = {"r": float(r), "p": float(p)}
        results["spearman"][pred] = {"rho": float(rho), "p": float(p_rho)}
        print(f"  {pred:>16}: r={r:+.3f} (p={p:.2e})  rho={rho:+.3f} (p={p_rho:.2e})")

    # --- Partial correlations conditioning on out_degree (the Yan-test) ---
    print("\n=== Partial r (controlling for out_degree) ===")
    z = df["out_degree"].to_numpy().astype(float)
    for pred in predictors:
        if pred == "out_degree":
            continue
        x = df[pred].to_numpy().astype(float)
        r, p = partial_corr(x, y, z)
        results["partial_vs_out_degree"][pred] = {"r": float(r), "p": float(p)}
        print(f"  {pred:>16} | out_degree: r={r:+.3f}  p={p:.2e}")

    # --- Run-level analysis: replicate the paper's r=0.97 finding and decompose ---
    print("\n=== Run-level (n=50,000) ===")
    rdf = pl.DataFrame(run_rows)
    y_cascade = rdf["cascade_size"].to_numpy().astype(float)
    y_removed = rdf["num_removed"].to_numpy().astype(float)
    ns_arr = rdf["num_seeds"].to_numpy().astype(float)

    results["run_level"] = {
        "n_runs": int(rdf.height),
        "pearson_vs_num_removed": {},
        "spearman_vs_num_removed": {},
        "pearson_vs_cascade_size": {},
        "spearman_vs_cascade_size": {},
        "partial_vs_num_seeds_target_cascade": {},
    }

    print("\nPaper's reported regression: num_removed vs sum_of_seed_predictor")
    for name in ["out_degree", "in_degree", "betweenness", "pagerank", "participation"]:
        x = rdf[f"sum_{name}"].to_numpy().astype(float)
        r, p = scistats.pearsonr(x, y_removed)
        rho, p_rho = scistats.spearmanr(x, y_removed)
        results["run_level"]["pearson_vs_num_removed"][f"sum_{name}"] = {
            "r": float(r), "p": float(p)
        }
        results["run_level"]["spearman_vs_num_removed"][f"sum_{name}"] = {
            "rho": float(rho), "p": float(p_rho)
        }
        print(f"  num_removed ~ sum_{name:>13}: r={r:+.3f}  rho={rho:+.3f}")

    print("\nStripped of seed-count confound: cascade_size vs mean_of_seed_predictor")
    for name in ["out_degree", "in_degree", "betweenness", "pagerank", "participation"]:
        x = rdf[f"mean_{name}"].to_numpy().astype(float)
        r, p = scistats.pearsonr(x, y_cascade)
        rho, p_rho = scistats.spearmanr(x, y_cascade)
        results["run_level"]["pearson_vs_cascade_size"][f"mean_{name}"] = {
            "r": float(r), "p": float(p)
        }
        results["run_level"]["spearman_vs_cascade_size"][f"mean_{name}"] = {
            "rho": float(rho), "p": float(p_rho)
        }
        print(f"  cascade_size ~ mean_{name:>13}: r={r:+.3f}  rho={rho:+.3f}")

    print("\nYan-test (run-level): partial r(cascade_size, mean_X | num_seeds)")
    for name in ["out_degree", "in_degree", "betweenness", "pagerank", "participation"]:
        x = rdf[f"mean_{name}"].to_numpy().astype(float)
        r, p = partial_corr(x, y_cascade, ns_arr)
        results["run_level"]["partial_vs_num_seeds_target_cascade"][f"mean_{name}"] = {
            "r": float(r), "p": float(p)
        }
        print(f"  cascade_size ~ mean_{name:>13} | num_seeds: r={r:+.3f}  p={p:.2e}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
