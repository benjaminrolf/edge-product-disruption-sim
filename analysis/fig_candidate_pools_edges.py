"""Figure 3: candidate pools of the products actually lost on failed edges.

The product lost on a failed edge is pi_E(u, v),
which is imputed uniformly from the portfolio on 93.7% of edges and is therefore
usually not the scarcest one. This script evaluates both rules on the Experiment-1
network configuration, one value per (edge, lost product), exactly as the
simulation builds the pools when supplier u fails on edge (u, v):
  - product-specific rule: firms offering a Sigma-similar product of p,
    excluding u and v (ImprovedZhaoNode.start_search),
  - portfolio-overlap rule: firms sharing >= 1 portfolio product with u,
    excluding u and v (OverlapZhaoNode.start_search).
The figure shows the cumulative share of lost edge products with at most x
candidates (log x-axis) for the overlap rule and for the product-specific rule on
imputed and on annotated edges; on the symlog axis, the value at x = 0 is the share
without any candidate. As in the runs of the article, firms can also supply the products
they are observed to deliver (delivered_products).

Usage: uv run python fig_candidate_pools_edges.py [--recompute]
  Without --recompute the figure is drawn from candidate_pools_edges.json.

Outputs: outputs/figures/fig3_candidate_pools_edges.pdf (+ candidate_pools_edges.json)
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import paper_style as st
from graphsim.networks.marklines import (
    create_comp_hash_new,
    get_delivered_products,
    load_marklines,
)
from graphsim.zhao.config import NetworkConfig
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.monte_carlo import create_inv_dists, load_edge_data
from graphsim.zhao.search import ImprovedSearch
from graphsim.zhao.wrappers import configure_network
from graphsim import paths

st.apply()

DATA_DIR = paths.DATA_DIR
EMBEDDINGS_DIR = paths.EMBEDDINGS_DIR
RESULTS_DIR = paths.CONFIGS_DIR
FIGDIR = paths.FIGURES_DIR
DATA_FILE = FIGDIR / "candidate_pools_edges.json"


def compute() -> None:
    net, data, id_firm_map = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch,
    )
    edge_weight_kde, product_kdes, edge_alternative_products = load_edge_data(
        data=data, edge_product_path=EMBEDDINGS_DIR / "top_edge_products.json",
    )
    product_product_map, product_firm_map = create_comp_hash_new(
        firms=id_firm_map.values(),
        path=EMBEDDINGS_DIR / "alternative_products.json",
        delivered_products=get_delivered_products(net),
    )
    del data

    # Experiment-1 network configuration (all runs share network_id 0)
    with open(RESULTS_DIR / "1_comp_configs_rev.json", encoding="utf-8") as f:
        nc = json.load(f)[0]["network_config"]
    net_config = NetworkConfig(
        network_seed=nc["network_seed"], dist_cap=nc["dist_cap"],
        dist_k=nc["dist_k"], max_trials=nc["max_trials"],
    )
    configure_network(
        net=net, id_firm_map=id_firm_map, net_config=net_config,
        edge_weight_kde=edge_weight_kde, product_kdes=product_kdes,
        edge_alternative_products=edge_alternative_products,
        inv_dists=create_inv_dists(net_config),
    )

    sigma_pool_cache: dict[str, set] = {}

    def sigma_pool(product: str) -> set:
        if product not in sigma_pool_cache:
            firms = set()
            for alt_product, _ in product_product_map[product]:
                firms.update(f.id for f in product_firm_map.get(alt_product, ()))
            sigma_pool_cache[product] = firms
        return sigma_pool_cache[product]

    overlap_pool_cache: dict = {}

    def overlap_pool(supplier_id) -> set:
        if supplier_id not in overlap_pool_cache:
            firms = set()
            for product in id_firm_map[supplier_id].products:
                firms.update(f.id for f in product_firm_map.get(product, ()))
            firms.discard(supplier_id)
            overlap_pool_cache[supplier_id] = firms
        return overlap_pool_cache[supplier_id]

    specific, overlap, annotated = [], [], []
    not_in_sigma = 0
    for u, v, d in net.edges(data=True):
        o_pool = overlap_pool(u)
        o_size = len(o_pool) - (v in o_pool)
        for product in d["products"]:
            if product not in product_product_map:
                not_in_sigma += 1
                continue
            s_pool = sigma_pool(product)
            specific.append(len(s_pool) - (u in s_pool) - (v in s_pool))
            overlap.append(o_size)
            annotated.append(not d["stochastic"])

    specific = np.array(specific)
    overlap = np.array(overlap)
    annotated = np.array(annotated)
    n_lost = specific.size + not_in_sigma
    empty = int((specific == 0).sum())
    print(f"lost edge products: {n_lost} "
          f"(not in Sigma map: {not_in_sigma}, empty pool: {empty} = "
          f"{100 * empty / n_lost:.2f}%)")
    for label, arr in (("product-specific", specific), ("overlap", overlap)):
        print(f"{label}: median={np.median(arr):.0f}, "
              f"q25={np.percentile(arr, 25):.0f}, share<=10="
              f"{100 * (arr <= 10).mean():.1f}%")
    print(f"product-specific on annotated edges only: median="
          f"{np.median(specific[annotated]):.0f} (n={annotated.sum()})")

    json.dump({
        "product_specific_per_edge_product": specific.tolist(),
        "overlap_per_edge_product": overlap.tolist(),
        "annotated_edge": annotated.tolist(),
        "not_in_sigma_map": not_in_sigma,
    }, open(DATA_FILE, "w"))


def ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Share of values <= each grid point."""
    return np.searchsorted(np.sort(values), grid, side="right") / values.size


def plot() -> None:
    with open(DATA_FILE) as f:
        d = json.load(f)
    specific = np.array(d["product_specific_per_edge_product"])
    overlap = np.array(d["overlap_per_edge_product"])
    annotated = np.array(d["annotated_edge"])

    series = (  # legend label, values, colour, direct label, its position
        ("Portfolio-overlap rule", overlap, st.ORANGE,
         "overlap", (2.4e4, 0.55, "right")),
        ("Product-specific rule, imputed edges", specific[~annotated], st.BLUE,
         "imputed edges", (150, 0.1, "left")),
        ("Product-specific rule, annotated edges", specific[annotated], st.GREEN,
         "annotated edges", (1.2, 0.38, "left")),
    )
    for label, values, *_ in series:
        print(f"{label}: n={values.size}, median={np.median(values):.0f}, "
              f"<=10: {100 * (values <= 10).mean():.1f}%, "
              f"none: {100 * (values == 0).mean():.2f}%")

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    # Linear below 1 so that x = 0 (no candidate) is shown, logarithmic above
    ax.set_xscale("symlog", linthresh=1, linscale=0.4)
    grid = np.concatenate([[0], np.unique(np.round(np.logspace(0, 5, 600)))])
    for label, values, color, direct, (x, y, ha) in series:
        ax.plot(grid, ecdf(values, grid), color=color, lw=1.6, label=label,
                drawstyle="steps-post")
        ax.text(x, y, direct, fontsize=7.5, color=st.INK, ha=ha, va="center")

    ax.axvline(10, color=st.GREY, lw=0.8, ls=":", zorder=0)
    ticks = [0, 1, 10, 100, 1_000, 10_000, 100_000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(0, 1e5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Substitution candidates x")
    ax.set_ylabel("Share of lost edge products\nwith at most x candidates")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False,
              fontsize=7.5, handlelength=1.5, columnspacing=1.2, borderaxespad=0)

    fig.savefig(FIGDIR / "fig3_candidate_pools_edges.pdf")
    print("saved fig3_candidate_pools_edges.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute the pools from the network (about 10 min).")
    args = parser.parse_args()
    if args.recompute or not DATA_FILE.exists():
        compute()
    plot()


if __name__ == "__main__":
    main()
