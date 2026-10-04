"""Data-based plausibility checks of the similarity rule across tau.

Neither check validates technical substitutability; both bound how the proxy errs.

1. Taxonomy check (false positives). MarkLines classifies its 963 standard products
   into 117 groups and 13 families. Among the Sigma pairs of two different standard
   products, the share in different families is a lower bound of false matches (for
   example a seat part matched with an engine part).
2. Dual-sourcing check (false negatives). When an OEM sources the same product for the
   same vehicle model and model year from two suppliers, each is a revealed substitute
   of the other. The share of such directed pairs (u1 -> u2, product p) in which the
   rule lists u2 as a candidate when u1 fails is the recall of the rule. Candidates
   are built as in the simulation (ImprovedZhaoNode): firms whose initial portfolio
   holds a product Sigma-similar to p. The portfolio-overlap rule is shown for
   comparison, as is the median candidate pool of the lost products (the cost of recall).
   The 'delivered' recall also counts u2 as a candidate if it is observed delivering a
   Sigma-similar product to a customer other than the one of the pair (out of sample),
   as in the runs of the article with delivered_products.

Output: substitutability_checks.json next to this script.
"""

import itertools
import json
from collections import defaultdict

import numpy as np
import polars as pl

from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

DATA_DIR = paths.DATA_DIR
EMBEDDINGS_DIR = paths.EMBEDDINGS_DIR
TAUS = [0.65, 0.69, 0.73, 0.77, 0.81, 0.85]


def revealed_pairs(keys: list[str]) -> list[tuple]:
    """Directed pairs (u1, u2, product, customer) of suppliers delivering the same product."""
    relations = pl.read_csv(DATA_DIR / "SupplyRelations.csv", infer_schema_length=10000)
    details = pl.read_csv(DATA_DIR / "SupplyRelationDetails.csv",
                          infer_schema_length=10000)
    joined = details.join(
        relations.select(["relation_id", "source_firm_id", "target_firm_id"]),
        on="relation_id",
    )
    groups = joined.group_by(keys + ["product_name"]).agg(
        pl.col("source_firm_id").unique().alias("suppliers"),
    ).filter(pl.col("suppliers").list.len() >= 2)
    pairs = set()
    for customer, product, suppliers in groups.select(
        ["target_firm_id", "product_name", "suppliers"],
    ).rows():
        for u1, u2 in itertools.permutations(sorted(suppliers), 2):
            pairs.add((u1, u2, product, customer))
    return sorted(pairs)


def main() -> None:
    net, _, id_firm_map = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch,
    )
    product_firms: dict[str, set] = defaultdict(set)
    for fid, firm in id_firm_map.items():
        for product in firm.products:
            product_firms[product].add(fid)

    taxonomy = pl.read_csv(DATA_DIR / "Products.csv", infer_schema_length=10000)
    group_of = dict(zip(taxonomy["product_name"], taxonomy["group_id"], strict=True))
    family_of = dict(zip(taxonomy["product_name"], taxonomy["family_id"], strict=True))
    standard = set(group_of)

    relations = pl.read_csv(DATA_DIR / "SupplyRelations.csv", infer_schema_length=10000)
    details = pl.read_csv(DATA_DIR / "SupplyRelationDetails.csv",
                          infer_schema_length=10000)
    to_customer = defaultdict(lambda: defaultdict(set))
    for u, v, p in details.join(relations, on="relation_id").select(
        ["source_firm_id", "target_firm_id", "product_name"],
    ).rows():
        to_customer[u][v].add(p)

    strict = [p for p in revealed_pairs(["target_firm_id", "model_name", "model_year"])
              if p[0] in id_firm_map and p[1] in id_firm_map]
    loose = [p for p in revealed_pairs(["target_firm_id"])
             if p[0] in id_firm_map and p[1] in id_firm_map]
    print(f"revealed directed pairs: strict {len(strict)}, loose {len(loose)}", flush=True)

    def overlap_recall(pairs) -> float:
        return float(np.mean([bool(id_firm_map[u1].products & id_firm_map[u2].products)
                              for u1, u2, _, _ in pairs]))

    results = {"n_strict": len(strict), "n_loose": len(loose),
               "overlap_recall_strict": overlap_recall(strict),
               "overlap_recall_loose": overlap_recall(loose), "by_tau": {}}
    print(f"portfolio-overlap rule recall: strict {results['overlap_recall_strict']:.3f}, "
          f"loose {results['overlap_recall_loose']:.3f}", flush=True)

    for tau in TAUS:
        with open(EMBEDDINGS_DIR / f"alternative_products_tau{tau}.json") as f:
            raw = json.load(f)
        similar = {p: set(v["products"]) for p, v in raw.items()}
        del raw

        # 1. Taxonomy: Sigma pairs of two different standard products
        same_group = same_family = total = 0
        for p in standard:
            for q in similar.get(p, ()):
                if q == p or q not in standard or q < p:
                    continue
                total += 1
                same_group += group_of[p] == group_of[q]
                same_family += family_of[p] == family_of[q]

        # 2. Dual sourcing: is the revealed substitute u2 a candidate for p?
        pool_cache: dict[str, set] = {}

        def pool(product: str) -> set:
            if product not in pool_cache:
                firms = set()
                for q in similar.get(product, {product}):
                    firms |= product_firms.get(q, set())
                pool_cache[product] = firms
            return pool_cache[product]

        def recall(pairs, delivered: bool = False) -> tuple[float, float]:
            hits = []
            for _, u2, p, v in pairs:
                hit = u2 in pool(p)
                if delivered and not hit:
                    elsewhere = set().union(*(prods for c, prods in to_customer[u2].items()
                                              if c != v))
                    hit = bool(similar.get(p, {p}) & elsewhere)
                hits.append(hit)
            pools = [len(pool(p)) for p in {p for _, _, p, _ in pairs}]
            return float(np.mean(hits)), float(np.median(pools))

        rs, pool_s = recall(strict)
        rl, _ = recall(loose)
        rds, _ = recall(strict, delivered=True)
        rdl, _ = recall(loose, delivered=True)
        res = {"taxonomy_pairs": total,
               "share_same_group": same_group / total if total else None,
               "share_other_family": 1 - same_family / total if total else None,
               "recall_strict": rs, "recall_loose": rl,
               "recall_strict_delivered": rds, "recall_loose_delivered": rdl,
               "median_pool_revealed_products": pool_s}
        results["by_tau"][str(tau)] = res
        print(f"tau {tau}: taxonomy pairs {total}, same group {100 * res['share_same_group']:.1f}%, "
              f"other family {100 * res['share_other_family']:.1f}% | dual-sourcing recall "
              f"strict {100 * rs:.1f}%, loose {100 * rl:.1f}%, with deliveries strict "
              f"{100 * rds:.1f}%, loose {100 * rdl:.1f}% | median pool {pool_s:.0f}",
              flush=True)
        del similar, pool_cache

    out = paths.ANALYSIS_DIR / "substitutability_checks.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
