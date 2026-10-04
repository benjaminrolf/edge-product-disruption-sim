"""What makes a seed firm critical beyond its out-degree.

Out-degree counts how many customers a failed supplier exposes. Whether an exposed
customer fails depends, in the model, on two further link properties:
  log_pool    how many firms can substitute the lost product (log of the Sigma candidate
              pool; annotated edges: the scarcest observed product, imputed edges: the
              mean over the supplier's portfolio, from which the imputation draws)
  dependence  the share of the customer's inbound volume that the link carries (imputed
              edges weighted with the mean annotated weight), which drives l(f)
and on the customer's size (log degree), which lowers the removal probability mu(f).

Link level: for every customer c of a seed firm s, the share of runs with s in the
seed set in which c is removed with s among its disruption origins. Logistic models
of this propagation probability are fitted with 5-fold cross-validation over seed
firms, from a constant rate (pure topology) via the customer's degree and dependence
to the full model with the product pool, so the gain from product data is separated
from the gain from looking further downstream.
Firm level: the mean attributed cascade of a seed firm (root-cause attribution as in
cascade_attribution.py) is compared with
  out_degree  the benchmark of the paper (Table D1)
  exposure1   expected number of directly removed customers, sum_c p(s, c)
  exposure3   expected cascade under independent propagation up to depth 3,
              R(s) = sum_c p(s, c) (1 + R(c)), with p from the fold's link model
by Pearson and Spearman correlation, partial correlation given out-degree and given
the topology-only exposure, and the overlap of the top-50 firms.
Streamed run counts are cached next to --out (.cache.pkl).

Usage:
  uv run python critical_firm_mechanism.py --runs DIR [--delivered-products] [--out FILE]
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import stats as scistats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cascade_attribution import attribute_run
from graphsim.networks.marklines import get_delivered_products, load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

DATA_DIR = paths.DATA_DIR
SIGMA_PATH = paths.EMBEDDINGS_DIR / "alternative_products.json"
FEATURES = ("log_pool", "dependence", "log_customer_degree")
MODELS = {
    "constant": (),
    "topology": ("log_customer_degree",),
    "volume": ("dependence", "log_customer_degree"),
    "full": FEATURES,
}
FOLDS = 5
DEPTH = 3
TOP = 50


def link_features(net, id_firm_map, delivered: bool) -> dict:
    """Feature vector per edge (source, target) of the base network."""
    with open(SIGMA_PATH) as f:
        similar = {p: v["products"] for p, v in json.load(f).items()}
    product_firms: dict[str, set] = defaultdict(set)
    for fid, firm in id_firm_map.items():
        for product in firm.products:
            product_firms[product].add(fid)
    if delivered:
        for fid, products in get_delivered_products(net).items():
            for product in products:
                product_firms[product].add(fid)

    pool_cache: dict[str, float] = {}

    def log_pool(product: str) -> float:
        if product not in pool_cache:
            firms = set()
            for q in similar.get(product, (product,)):
                firms |= product_firms.get(q, set())
            pool_cache[product] = float(np.log(max(len(firms), 1)))
        return pool_cache[product]

    portfolio_pool = {fid: float(np.mean([log_pool(p) for p in sorted(firm.products)]))
                      for fid, firm in id_firm_map.items() if firm.products}
    annotated = [d["weight"] for _, _, d in net.edges(data=True) if not d["stochastic"]]
    mean_weight = float(np.mean(annotated))

    def weight(data) -> float:
        return mean_weight if data["stochastic"] else float(data["weight"])

    inbound = {c: sum(weight(d) for _, _, d in net.in_edges(c, data=True))
               for c in net.nodes}
    features = {}
    for s, c, d in net.edges(data=True):
        if d["stochastic"]:
            pool = portfolio_pool.get(s)
        else:
            pool = min((log_pool(p) for p in d["products"]), default=None)
        if pool is None:
            continue
        features[(s, c)] = (pool, weight(d) / inbound[c], float(np.log(net.degree(c))))
    return features


def stream_runs(runs_dir: Path, net) -> tuple:
    """Per-link exposure/propagation counts and per-seed attributed cascades."""
    seed_sets: dict[frozenset, int] = defaultdict(int)
    propagated: dict[tuple, int] = defaultdict(int)
    mass_sum: dict[int, float] = defaultdict(float)
    mass_n: dict[int, int] = defaultdict(int)
    n_runs = 0
    files = sorted(runs_dir.glob("stats_*.jsonl"))
    for i, path in enumerate(files, 1):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                seeds = frozenset(s["id"] for s in rec["seeds"])
                seed_sets[seeds] += 1
                for r in rec["removed"]:
                    if r["id"] in seeds:
                        continue
                    for o in set(r.get("disruption_origin") or ()):
                        if o in seeds:
                            propagated[(o, r["id"])] += 1
                mass, *_ = attribute_run(rec)
                for s, v in mass.items():
                    mass_sum[s] += v
                    mass_n[s] += 1
                n_runs += 1
        if i % 50 == 0:
            print(f"  {i}/{len(files)} files, {n_runs} runs", flush=True)
    # A customer is exposed in every run in which its supplier is a seed and it is not
    exposed: dict[tuple, int] = defaultdict(int)
    for seeds, count in seed_sets.items():
        for s in seeds:
            if s not in net:
                continue
            for c in net.successors(s):
                if c not in seeds:
                    exposed[(s, c)] += count
    attributed = {s: mass_sum[s] / mass_n[s] for s in mass_sum}
    return exposed, propagated, attributed, n_runs


def fit_link_model(X: np.ndarray, k: np.ndarray, n: np.ndarray) -> LogisticRegression:
    """Binomial logistic regression on aggregated counts (successes and failures)."""
    Xs = np.vstack([X, X])
    ys = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    ws = np.concatenate([k, n - k])
    keep = ws > 0
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(Xs[keep], ys[keep], sample_weight=ws[keep])
    return model


def standardise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    def resid(a):
        slope, intercept, *_ = scistats.linregress(z, a)
        return a - (slope * z + intercept)
    return float(scistats.pearsonr(resid(x), resid(y))[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--delivered-products", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=paths.ANALYSIS_DIR / "critical_firm_mechanism_exp2.json")
    args = parser.parse_args()

    print("Loading network and link features ...", flush=True)
    net, _, id_firm_map = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch,
    )
    features = link_features(net, id_firm_map, args.delivered_products)
    print(f"  {len(features)} of {net.number_of_edges()} edges with features", flush=True)

    cache = args.out.with_suffix(".cache.pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            exposed, propagated, attributed, n_runs = pickle.load(f)
    else:
        print("Streaming runs ...", flush=True)
        exposed, propagated, attributed, n_runs = stream_runs(args.runs, net)
        with open(cache, "wb") as f:
            pickle.dump((exposed, propagated, attributed, n_runs), f)
    links = sorted(link for link, n in exposed.items() if n > 0 and link in features)
    X = np.array([features[link] for link in links])
    n = np.array([exposed[link] for link in links], dtype=float)
    k = np.array([propagated.get(link, 0) for link in links], dtype=float)
    base_rate = k.sum() / n.sum()
    print(f"runs {n_runs}, seed links {len(links)}, exposures {n.sum():.0f}, "
          f"propagation rate {100 * base_rate:.2f}%", flush=True)

    mean, std = X.mean(axis=0), X.std(axis=0)
    Xz = standardise(X, mean, std)
    seed_firms = sorted({s for s, _ in links})
    fold_of = {s: i % FOLDS for i, s in enumerate(
        np.random.default_rng(0).permutation(seed_firms))}
    link_fold = np.array([fold_of[s] for s, _ in links])
    y_bin = np.concatenate([np.ones(len(links)), np.zeros(len(links))])
    w_bin = np.concatenate([k, n - k])

    all_links = sorted(features)
    nodes = sorted(net.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    rows = np.array([index[s] for s, _ in all_links])
    cols = np.array([index[c] for _, c in all_links])
    Xall = standardise(np.array([features[link] for link in all_links]), mean, std)

    def exposure(p_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = sp.csr_matrix((p_all, (rows, cols)), shape=(len(nodes), len(nodes)))
        R = P @ np.ones(len(nodes))
        R1 = R.copy()
        for _ in range(DEPTH - 1):
            R = P @ (1.0 + R)
        return R1, R

    # Link models from topology alone to the full model with the product pool; the
    # constant model propagates with the base rate on every link (pure downstream reach)
    link_models, exposures = {}, {}
    for variant, cols_used in MODELS.items():
        j = [FEATURES.index(f) for f in cols_used]
        p_cv = np.zeros(len(links))
        e1, e3 = {}, {}
        for fold in range(FOLDS):
            train = link_fold != fold
            if j:
                model = fit_link_model(Xz[train][:, j], k[train], n[train])
                p_cv[~train] = model.predict_proba(Xz[~train][:, j])[:, 1]
                p_all = model.predict_proba(Xall[:, j])[:, 1]
            else:
                rate = k[train].sum() / n[train].sum()
                p_cv[~train] = rate
                p_all = np.full(len(all_links), rate)
            R1, R3 = exposure(p_all)
            for s in seed_firms:
                if fold_of[s] == fold:
                    e1[s], e3[s] = float(R1[index[s]]), float(R3[index[s]])
        exposures[variant] = (e1, e3)
        entry = {"auc_cv": (roc_auc_score(y_bin, np.concatenate([p_cv, p_cv]),
                                          sample_weight=w_bin) if j else 0.5)}
        if j:
            full_fit = fit_link_model(Xz[:, j], k, n)
            entry["standardised_coefficients"] = dict(
                zip(cols_used, full_fit.coef_[0].tolist(), strict=True))
        link_models[variant] = entry

    by_quartile = {}
    for j, name in enumerate(FEATURES):
        edges = np.quantile(X[:, j], [0.25, 0.5, 0.75])
        q = np.searchsorted(edges, X[:, j], side="right")
        by_quartile[name] = [float(k[q == i].sum() / n[q == i].sum()) for i in range(4)]

    # Firm level
    firms = [s for s in seed_firms if s in attributed]
    y = np.array([attributed[s] for s in firms])
    predictors = {"out_degree": np.array([net.out_degree(s) for s in firms], dtype=float)}
    for variant, (e1, e3) in exposures.items():
        predictors[f"{variant}_exposure1"] = np.array([e1[s] for s in firms])
        predictors[f"{variant}_exposure{DEPTH}"] = np.array([e3[s] for s in firms])
    top_true = set(np.argsort(-y)[:TOP])
    topo = predictors[f"topology_exposure{DEPTH}"]
    firm_level = {}
    for name, x in predictors.items():
        firm_level[name] = {
            "pearson": float(scistats.pearsonr(x, y)[0]),
            "pearson_log": float(scistats.pearsonr(np.log1p(x), np.log1p(y))[0]),
            "spearman": float(scistats.spearmanr(x, y)[0]),
            "partial_given_out_degree": partial_r(x, y, predictors["out_degree"]),
            f"partial_given_topology_exposure{DEPTH}": partial_r(x, y, topo),
            f"top{TOP}_overlap": len(top_true & set(np.argsort(-x)[:TOP])) / TOP,
        }

    results = {
        "runs": n_runs, "seed_firms": len(firms), "seed_links": len(links),
        "propagation_rate": base_rate, "link_models": link_models,
        "univariate_auc": {
            name: roc_auc_score(y_bin, np.concatenate([Xz[:, j], Xz[:, j]]),
                                sample_weight=w_bin)
            for j, name in enumerate(FEATURES)},
        "propagation_rate_by_quartile": by_quartile,
        "firm_level": firm_level,
    }
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
