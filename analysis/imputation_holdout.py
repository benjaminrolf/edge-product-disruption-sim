"""Hold-out test of the edge-product imputation.

For every annotated edge (u, v), the observed products are hidden and replaced by
the imputation of Section 4.1.3. The observed number of products k is kept, so that
only the choice of products is tested: k products are drawn without replacement
from the supplier's portfolio pi_V(u), uniformly (gamma = 0, the current method) or
with probability proportional to (pool + 1)^gamma. Here pool is the number of
substitution candidates of a product as the simulation builds it: firms offering a
Sigma-similar product, excluding u and v. The imputation draws from the initial
portfolios, to which annotated edge products are added only after sampling, so the
hidden products need no further masking.

Reported per gamma (R draws per edge):
  - recall: share of observed products with a Sigma-similar (or identical) imputed
    product on the same edge
  - substitution difficulty: pool sizes of imputed vs observed products (median,
    share <= 10, share without any candidate), also for the scarcest product per
    edge, and the Spearman correlation of the scarcest pool across edges
  - KS distance between the log pool distributions of imputed and observed products;
    gamma* minimises it, validated by 5-fold cross-validation over edges
Also reports the tier composition of annotated vs unannotated edges, since the
observed annotations are biased towards Tier-1-to-OEM links, and the pools of
observed products whose name also occurs in the portfolio vocabulary, to separate
scarcity from vocabulary differences between edge and portfolio product names.

With --delivered, pools also count firms observed delivering a similar product
(delivered_products, as in the runs of the article).

Output: imputation_holdout[_delivered].json next to this script.
"""

import argparse
import json

import numpy as np
from scipy.stats import ks_2samp, spearmanr

from graphsim.networks.marklines import (
    create_comp_hash_new,
    get_delivered_products,
    load_marklines,
)
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

DATA_DIR = paths.DATA_DIR
EMBEDDINGS_DIR = paths.EMBEDDINGS_DIR
GAMMAS = [-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
DRAWS = 20
FOLDS = 5
SEED = 42


def summarise(pools: np.ndarray) -> dict:
    return {
        "median": float(np.median(pools)),
        "share_le10": float((pools <= 10).mean()),
        "share_zero": float((pools == 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivered", action="store_true")
    args = parser.parse_args()

    net, _, id_firm_map = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch,
    )
    product_product_map, product_firm_map = create_comp_hash_new(
        firms=id_firm_map.values(),
        path=EMBEDDINGS_DIR / "alternative_products.json",
        delivered_products=get_delivered_products(net) if args.delivered else None,
    )

    similar = {p: {q for q, _ in alts} for p, alts in product_product_map.items()}
    pool_cache: dict[str, set] = {}

    def pool_set(product: str) -> set:
        if product not in pool_cache:
            firms = set()
            for alt in similar.get(product, ()):
                firms.update(f.id for f in product_firm_map.get(alt, ()))
            pool_cache[product] = firms
        return pool_cache[product]

    def pool_size(product: str, u, v) -> int:
        s = pool_set(product)
        return len(s) - (u in s) - (v in s)

    # Annotated edges whose supplier has a portfolio to impute from
    edges, skipped = [], 0
    for u, v, d in net.edges(data=True):
        if d["stochastic"]:
            continue
        observed = list(d["products"])
        portfolio = sorted(id_firm_map[u].products)
        if not observed or not portfolio:
            skipped += 1
            continue
        k = min(len(observed), len(portfolio))
        port_pools = np.array([pool_size(q, u, v) for q in portfolio])
        obs_pools = np.array([pool_size(p, u, v) for p in observed])
        edges.append({"u": u, "v": v, "observed": observed, "k": k,
                      "portfolio": portfolio, "port_pools": port_pools,
                      "obs_pools": obs_pools})
    print(f"annotated edges used: {len(edges)} (skipped without products or "
          f"portfolio: {skipped})", flush=True)

    # Tier composition: annotated vs unannotated edges (target tier 0 = OEM / sink)
    def tier_share(stochastic: bool) -> dict:
        tiers = [id_firm_map[v].tier for u, v, d in net.edges(data=True)
                 if d["stochastic"] == stochastic]
        tiers = np.array(tiers)
        return {"n": int(tiers.size), "to_tier0": float((tiers == 0).mean()),
                "to_tier1": float((tiers == 1).mean()),
                "to_tier2plus": float((tiers >= 2).mean())}
    composition = {"annotated": tier_share(False), "unannotated": tier_share(True)}
    print("edge composition by customer tier:", composition, flush=True)

    rng = np.random.default_rng(SEED)
    fold_of = rng.integers(0, FOLDS, size=len(edges))
    obs_all = np.concatenate([e["obs_pools"] for e in edges])
    obs_min = np.array([e["obs_pools"].min() for e in edges])

    results = {}
    imputed_logs = {}  # gamma -> per-edge list of imputed log pools (for CV)
    for gamma in GAMMAS:
        rng = np.random.default_rng(SEED)
        imp_all, recall_hits, recall_n, imp_min_med = [], 0, 0, []
        per_edge_logs = []
        for e in edges:
            weights = (e["port_pools"] + 1.0) ** gamma
            weights = weights / weights.sum()
            n_nonzero = int((weights > 0).sum())
            k = min(e["k"], n_nonzero)
            observed_similar = [similar.get(p, {p}) | {p} for p in e["observed"]]
            mins, logs = [], []
            for _ in range(DRAWS):
                idx = rng.choice(len(e["portfolio"]), size=k, replace=False, p=weights)
                chosen = [e["portfolio"][i] for i in idx]
                pools = e["port_pools"][idx]
                imp_all.append(pools)
                logs.append(np.log1p(pools))
                mins.append(pools.min())
                chosen_set = set(chosen)
                recall_hits += sum(bool(s & chosen_set) for s in observed_similar)
                recall_n += len(observed_similar)
            imp_min_med.append(np.median(mins))
            per_edge_logs.append(np.concatenate(logs))
        imp_all = np.concatenate(imp_all)
        imputed_logs[gamma] = per_edge_logs
        res = {
            "recall_similar": recall_hits / recall_n,
            "imputed": summarise(imp_all),
            "imputed_scarcest": summarise(np.array(imp_min_med)),
            "spearman_scarcest": float(spearmanr(obs_min, imp_min_med).statistic),
            "ks_log_pool": float(ks_2samp(np.log1p(obs_all), np.log1p(imp_all)).statistic),
        }
        results[str(gamma)] = res
        print(f"gamma {gamma:+.2f}: recall {res['recall_similar']:.3f}, "
              f"imputed median {res['imputed']['median']:.0f}, "
              f"<=10 {100 * res['imputed']['share_le10']:.1f}%, "
              f"none {100 * res['imputed']['share_zero']:.1f}%, "
              f"scarcest median {res['imputed_scarcest']['median']:.0f}, "
              f"spearman {res['spearman_scarcest']:.2f}, KS {res['ks_log_pool']:.3f}",
              flush=True)

    # Vocabulary check: observed products that are also portfolio product names
    portfolio_vocab = set().union(*(f.products for f in id_firm_map.values()))
    in_vocab = np.concatenate([[p in portfolio_vocab for p in e["observed"]]
                               for e in edges])
    observed = {"all": summarise(obs_all), "scarcest": summarise(obs_min),
                "share_in_portfolio_vocab": float(in_vocab.mean()),
                "in_portfolio_vocab": summarise(obs_all[in_vocab]),
                "not_in_portfolio_vocab": summarise(obs_all[~in_vocab])}
    print(f"observed: {observed}", flush=True)

    # 5-fold CV of gamma*: choose on 4 folds (min KS), evaluate on the held-out fold
    obs_logs = [np.log1p(e["obs_pools"]) for e in edges]
    cv = []
    for f in range(FOLDS):
        train = np.where(fold_of != f)[0]
        test = np.where(fold_of == f)[0]
        def ks(gamma, idx):
            imp = np.concatenate([imputed_logs[gamma][i] for i in idx])
            obs = np.concatenate([obs_logs[i] for i in idx])
            return ks_2samp(obs, imp).statistic
        best = min(GAMMAS, key=lambda g: ks(g, train))
        cv.append({"fold": f, "gamma_star": best, "ks_test_star": float(ks(best, test)),
                   "ks_test_uniform": float(ks(0.0, test))})
    print("5-fold CV:", cv, flush=True)

    suffix = "_delivered" if args.delivered else ""
    out = paths.ANALYSIS_DIR / f"imputation_holdout{suffix}.json"
    out.write_text(json.dumps({
        "edges": len(edges), "draws_per_edge": DRAWS, "composition": composition,
        "observed": observed, "by_gamma": results, "cv": cv,
    }, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
