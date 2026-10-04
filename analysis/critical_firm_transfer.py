"""Does the criticality index from random shocks carry over to the regional shock?

The link models of critical_firm_mechanism.py are fitted on the Experiment-2 runs
(random seeds) and applied without refitting to the Tohoku-inspired runs (Experiment 3),
whose seeds are spatially concentrated and fail jointly. Two tests:
  criticality  mean attributed cascade of each Tohoku seed firm (root-cause attribution)
               against out-degree and the depth-3 exposure index of each link model;
               firms that were Experiment-2 seeds are excluded, so firms and shock are
               both out of sample
  reach        removal frequency of the firms that are never seeded (OEMs, firms
               outside Japan and outside the geocoded population) but have a supplier
               that fails in at least one run, against the removal probability
               predicted by independent propagation from the observed seed
               frequencies, q(c) = 1 - (1 - pi(c)) prod_s (1 - q(s) p(s, c)), depth 3
For both, the constant and topology-only link models show what topology alone predicts.

Usage:
  uv run python critical_firm_transfer.py --exp2-cache FILE --runs DIR [--out FILE]
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import stats as scistats
from sklearn.metrics import roc_auc_score

from cascade_attribution import attribute_run
from critical_firm_mechanism import (
    DATA_DIR,
    DEPTH,
    FEATURES,
    MODELS,
    TOP,
    fit_link_model,
    link_features,
    partial_r,
    standardise,
)
from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

MIN_SEED_RUNS = 100


def stream_regional(runs_dir: Path) -> tuple:
    seed_count: dict[int, int] = defaultdict(int)
    removed_count: dict[int, int] = defaultdict(int)
    mass_sum: dict[int, float] = defaultdict(float)
    n_runs = 0
    files = sorted(runs_dir.glob("stats_*.jsonl"))
    for i, path in enumerate(files, 1):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                seeds = {s["id"] for s in rec["seeds"]}
                for s in seeds:
                    seed_count[s] += 1
                for r in rec["removed"]:
                    if r["id"] not in seeds:
                        removed_count[r["id"]] += 1
                mass, *_ = attribute_run(rec)
                for s, v in mass.items():
                    mass_sum[s] += v
                n_runs += 1
        if i % 20 == 0:
            print(f"  {i}/{len(files)} files, {n_runs} runs", flush=True)
    return seed_count, removed_count, mass_sum, n_runs


def metrics(x: np.ndarray, y: np.ndarray, topo: np.ndarray) -> dict:
    top_true = set(np.argsort(-y)[:TOP])
    return {
        "pearson": float(scistats.pearsonr(x, y)[0]),
        "pearson_log": float(scistats.pearsonr(np.log1p(x), np.log1p(y))[0]),
        "spearman": float(scistats.spearmanr(x, y)[0]),
        "partial_given_topology": partial_r(x, y, topo),
        f"top{TOP}_overlap": len(top_true & set(np.argsort(-x)[:TOP])) / TOP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exp2-cache", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--delivered-products", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=paths.ANALYSIS_DIR / "critical_firm_transfer.json")
    args = parser.parse_args()

    net, _, id_firm_map = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch,
    )
    features = link_features(net, id_firm_map, args.delivered_products)
    with open(args.exp2_cache, "rb") as f:
        exposed, propagated, _, _ = pickle.load(f)
    links = sorted(link for link, n in exposed.items() if n > 0 and link in features)
    exp2_seeds = {s for s, _ in links}
    X = np.array([features[link] for link in links])
    n = np.array([exposed[link] for link in links], dtype=float)
    k = np.array([propagated.get(link, 0) for link in links], dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    Xz = standardise(X, mean, std)

    all_links = sorted(features)
    nodes = sorted(net.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    rows = np.array([index[s] for s, _ in all_links])
    cols = np.array([index[c] for _, c in all_links])
    Xall = standardise(np.array([features[link] for link in all_links]), mean, std)
    p_links = {}
    for variant, cols_used in MODELS.items():
        j = [FEATURES.index(f) for f in cols_used]
        if j:
            p_links[variant] = fit_link_model(Xz[:, j], k, n).predict_proba(Xall[:, j])[:, 1]
        else:
            p_links[variant] = np.full(len(all_links), k.sum() / n.sum())

    cache = args.out.with_suffix(".cache.pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            seed_count, removed_count, mass_sum, n_runs = pickle.load(f)
    else:
        print("Streaming regional runs ...", flush=True)
        seed_count, removed_count, mass_sum, n_runs = stream_regional(args.runs)
        with open(cache, "wb") as f:
            pickle.dump((seed_count, removed_count, mass_sum, n_runs), f)
    print(f"runs {n_runs}, seed firms {len(seed_count)}", flush=True)

    # Test 1: criticality of the regional seed firms
    firms = sorted(s for s, c in seed_count.items()
                   if c >= MIN_SEED_RUNS and s not in exp2_seeds and s in index)
    y = np.array([mass_sum[s] / seed_count[s] for s in firms])
    predictors = {"out_degree": np.array([net.out_degree(s) for s in firms], dtype=float)}
    for variant, p_all in p_links.items():
        P = sp.csr_matrix((p_all, (rows, cols)), shape=(len(nodes), len(nodes)))
        R = P @ np.ones(len(nodes))
        for _ in range(DEPTH - 1):
            R = P @ (1.0 + R)
        predictors[f"{variant}_exposure{DEPTH}"] = R[[index[s] for s in firms]]
    topo = predictors[f"topology_exposure{DEPTH}"]
    criticality = {name: metrics(x, y, topo) for name, x in predictors.items()}

    # Test 2: where the regional shock reaches, for firms that are never seeded
    pi = np.zeros(len(nodes))
    for s, c in seed_count.items():
        if s in index:
            pi[index[s]] = c / n_runs
    observed = np.array([removed_count.get(v, 0) / n_runs for v in nodes])
    # Firms far from the shock are trivially never removed, so only firms with a
    # supplier that fails in at least one run are evaluated
    failed = ((pi > 0) | (observed > 0)).astype(float)
    exposed = np.bincount(cols, weights=failed[rows], minlength=len(nodes)) > 0
    never = (pi == 0) & exposed
    reach = {"firms": int(never.sum()),
             "firms_ever_removed": int((never & (observed > 0)).sum())}
    predicted = {}
    for variant, p_all in p_links.items():
        q = pi.copy()
        for _ in range(DEPTH):
            log_survive = np.bincount(cols, weights=np.log1p(-q[rows] * p_all),
                                      minlength=len(nodes))
            q = 1.0 - (1.0 - pi) * np.exp(log_survive)
        predicted[variant] = q[never]
    obs = observed[never]
    topo_q = predicted["topology"]
    for variant, q in predicted.items():
        reach[variant] = {
            "auc_ever_removed": float(roc_auc_score(obs > 0, q)),
            **metrics(q, obs, topo_q),
        }

    results = {"runs": n_runs, "criticality_firms": len(firms),
               "criticality": criticality, "reach": reach}
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
