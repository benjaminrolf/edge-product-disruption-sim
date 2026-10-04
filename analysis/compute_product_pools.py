"""Pool size per product for the weighted edge-product imputation.

The pool of a product q is the number of firms offering a Sigma-similar product,
built as in ImprovedZhaoNode.start_search (alternative_products.json, the firms'
initial portfolios and the products they are observed to deliver, as in the
runs of the article with delivered_products). The supplier of an edge always offers its
own portfolio product, so pool - 1 is the number of candidates the simulation would
find (before excluding the customer). get_edge_products weights portfolio products with pool^gamma.

Output: product_pools.json in the embeddings directory.
"""

import json

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


def main() -> None:
    net, _, id_firm_map = load_marklines(
        DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch,
    )
    product_product_map, product_firm_map = create_comp_hash_new(
        firms=id_firm_map.values(),
        path=EMBEDDINGS_DIR / "alternative_products.json",
        delivered_products=get_delivered_products(net),
    )
    pools = {}
    for product, alternatives in product_product_map.items():
        firms = set()
        for alt, _ in alternatives:
            firms.update(f.id for f in product_firm_map.get(alt, ()))
        pools[product] = len(firms)

    out = EMBEDDINGS_DIR / "product_pools.json"
    out.write_text(json.dumps(pools, sort_keys=True))
    print(f"{len(pools)} products, saved {out}")


if __name__ == "__main__":
    main()
