"""Compute the product-substitutability maps for the tau sensitivity sweep.

The substitution map Sigma depends on the embedding-similarity threshold
tau = 0.73. For the sensitivity runs, this script computes the pairwise
similarities ONCE with the loosest threshold (0.65) and derives the
maps for all higher thresholds by filtering, producing
alternative_products_tau{0.65,0.69,0.73,0.77,0.81,0.85}.json next to the
canonical alternative_products.json.

Validation: the derived tau=0.73 map must reproduce the canonical map's pair
count (387,415 pairs over 45,446 products); the script prints the comparison.
top_edge_products.json is threshold-independent and is not touched.

The sweep configurations (experiments/revision_configs.py) select a map via
network_config["sigma_map"], and experiments/1_comparison_rev.py runs them.
"""

import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphsim.networks.marklines import load_product_embeddings
from graphsim import paths

EMBEDDINGS_DIR = paths.EMBEDDINGS_DIR

BASE_TAU = 0.65
TAUS = [0.65, 0.69, 0.73, 0.77, 0.81, 0.85]
BATCH = 512


def main() -> None:
    product_embeddings, edge_product_embeddings = load_product_embeddings(
        product_path=EMBEDDINGS_DIR / "product_embeddings.jsonl",
        edge_product_path=EMBEDDINGS_DIR / "edge_product_embeddings.jsonl",
    )
    all_products = list(product_embeddings.keys()) + list(edge_product_embeddings.keys())
    all_embeddings = np.array(
        list(product_embeddings.values()) + list(edge_product_embeddings.values()),
        dtype=np.float32,
    )
    n = len(all_products)
    print(f"{n} products, embedding dim {all_embeddings.shape[1]}")

    base: dict[str, dict] = {}
    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n)
        sims = cosine_similarity(all_embeddings[start:stop], all_embeddings)
        for row, i in enumerate(range(start, stop)):
            idx = np.where(sims[row] > BASE_TAU)[0]
            base[all_products[i]] = {
                "products": [all_products[j] for j in idx],
                "scores": sims[row][idx].astype(float).tolist(),
            }
        if (start // BATCH) % 10 == 0:
            print(f"  {stop}/{n}")

    for tau in TAUS:
        if tau == BASE_TAU:
            filtered = base
        else:
            filtered = {}
            for k, v in base.items():
                keep = [j for j, s in enumerate(v["scores"]) if s > tau]
                filtered[k] = {
                    "products": [v["products"][j] for j in keep],
                    "scores": [v["scores"][j] for j in keep],
                }
        pairs = sum(len(v["scores"]) for v in filtered.values())
        out = EMBEDDINGS_DIR / f"alternative_products_tau{tau}.json"
        with open(out, "w") as f:
            json.dump(filtered, f, indent=4, sort_keys=True)
        print(f"tau={tau}: {pairs} pairs -> {out.name}")

    # Validation against the canonical 0.73 map
    with open(EMBEDDINGS_DIR / "alternative_products.json") as f:
        canonical = json.load(f)
    can_pairs = sum(len(v["scores"]) for v in canonical.values())
    with open(EMBEDDINGS_DIR / "alternative_products_tau0.73.json") as f:
        derived = json.load(f)
    der_pairs = sum(len(v["scores"]) for v in derived.values())
    print(f"VALIDATION tau=0.73: canonical {can_pairs} pairs vs derived {der_pairs} pairs "
          f"({'MATCH' if can_pairs == der_pairs else 'MISMATCH — investigate'})")


if __name__ == "__main__":
    main()
