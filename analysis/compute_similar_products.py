"""Script to compute similar products based on their embeddings."""

import json

import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity

from graphsim.networks.marklines import load_product_embeddings
from graphsim import paths


def precompute_alternative_products_batched(
    product_embeddings: dict[str, npt.NDArray[np.float32]],
    edge_product_embeddings: dict[str, npt.NDArray[np.float32]],
    similarity_threshold: float = 0.73,
) -> dict[str, list[tuple[str, float]]]:
    """Find the most similar products by computing similarities in batches."""
    all_products = np.array(
        list(product_embeddings.keys())
        + list(
            edge_product_embeddings.keys(),
        ),
    )
    all_types = np.array(
        ["product"] * len(product_embeddings)
        + ["edge_product"]
        * len(
            edge_product_embeddings,
        ),
    )
    all_embeddings = np.array(
        list(product_embeddings.values()) + list(edge_product_embeddings.values()),
        dtype=np.float32,
    )

    alternative_products = {}
    top_edge_products = {}
    # Loop through each product to compute its similarities individually
    for i, product in enumerate(all_products):
        # Print progress every 100 products
        if i % 100 == 0:
            print(f"Processing product {i + 1}/{len(all_products)}")

        # Get the current product's embedding
        current_embedding = all_embeddings[i : i + 1]  # Shape: (1, D)

        # Compute similarity of this one embedding against all others
        # This creates a small (1, N) matrix, which is memory-efficient
        similarity_scores = cosine_similarity(current_embedding, all_embeddings)[0]

        # Get all indices where the similarity is above the threshold
        similar_indices = np.where(similarity_scores > similarity_threshold)[0]

        # Create lists of similar products and scores
        similar_items = {
            "products": all_products[similar_indices].tolist(),
            "scores": similarity_scores[similar_indices].tolist(),
        }
        alternative_products[product] = similar_items

        # For products get top edge product match (can be below threshold)
        if all_types[i] == "product":
            # Get the indices of edge products
            edge_indices = [
                idx for idx, t in enumerate(all_types) if t == "edge_product"
            ]
            # Get the similarity scores for edge products
            edge_scores = similarity_scores[edge_indices]
            # Find the index of the maximum score
            max_edge_index = np.argmax(edge_scores)
            # Get the corresponding edge product
            top_edge_products[product] = {
                "products": all_products[edge_indices[max_edge_index]],
                "scores": float(edge_scores[max_edge_index]),
            }

    return alternative_products, top_edge_products


if __name__ == "__main__":
    EMBEDDINGS_DIR = paths.EMBEDDINGS_DIR

    # Example usage
    product_embeddings, edge_product_embeddings = load_product_embeddings(
        product_path=EMBEDDINGS_DIR / "product_embeddings.jsonl",
        edge_product_path=EMBEDDINGS_DIR / "edge_product_embeddings.jsonl",
    )
    alternatives, top_edge_products = precompute_alternative_products_batched(
        product_embeddings,
        edge_product_embeddings,
    )

    # Save results to a JSON file
    output_path = EMBEDDINGS_DIR / "alternative_products.json"
    with open(output_path, "w") as f:
        json.dump(alternatives, f, indent=4, sort_keys=True)

    with open(EMBEDDINGS_DIR / "top_edge_products.json", "w") as f:
        json.dump(top_edge_products, f, indent=4, sort_keys=True)
