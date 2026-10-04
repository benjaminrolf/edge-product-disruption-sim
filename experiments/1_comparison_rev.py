"""Experiment-1 runner for the extended and the hybrid arm.

The arm is a CLI parameter, so that both arms share one code path and differ only in the node class, i.e. in the substitution rule:

  --arm extended   ImprovedZhaoNode: product-specific rule (embedding similarity)
  --arm hybrid     OverlapZhaoNode: portfolio-overlap rule of Zhao et al. (2019)

Both arms read 1_comp_configs_rev.json (paper calibration, seeds identical to the
baseline arm) and write to results/exp1_<arm>/. The substitution map Sigma is
network_config["sigma_map"] in embeddings/ (default alternative_products.json, tau =
0.73); the tau sweep sets it to alternative_products_tau<tau>.json.
"""

import argparse
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path

from graphsim.networks.marklines import (
    create_comp_hash_new,
    get_delivered_products,
    load_marklines,
)
from graphsim.zhao.config import NetworkConfig, SimulationConfig
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.monte_carlo import create_inv_dists, load_edge_data
from graphsim.zhao.overlap_node import OverlapZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim.zhao.wrappers import configure_network, simulate_zhao

NODE_CLASSES = {"extended": ImprovedZhaoNode, "hybrid": OverlapZhaoNode}
DEFAULT_SIGMA_MAP = "alternative_products.json"


def run_single_simulation_worker(
    config: dict,
    base_net,
    id_firm_map,
    comp_hash,
    edge_weight_kde,
    product_kdes,
    edge_alternative_products,
    product_pools=None,
):
    """A worker function that runs one simulation.

    Designed to be called by a multiprocessing Pool. It receives the large,
    pre-loaded data objects as arguments.

    Args:
        task_id: The unique ID for this simulation run.
        base_net: The master networkx graph object.
        (and other large data objects)...
        possible_seeds: List of node IDs that can be disrupted.

    Returns:
        The completed simulation model object.
    """
    start_time = time.time()

    # Unpack configs
    task_id = config["task_id"]
    sim_config = config["simulation_config"]
    net_config = config["network_config"]

    logging.info(
        f"Worker {os.getpid()} (Task {task_id}, Sample: {sim_config['sample_id']} "
        f"Network: {sim_config['network_id']}) "
    )

    # 2. Create a shallow copy of the network FOR THIS RUN ONLY
    net_copy = base_net.copy()
    diameter = 52

    # 3. Configure the network copy
    net_config = NetworkConfig(
        network_seed=net_config["network_seed"],
        dist_cap=net_config["dist_cap"],
        dist_k=net_config["dist_k"],
        max_trials=net_config["max_trials"],
        product_weight_exponent=net_config.get("product_weight_exponent", 0.0),
    )

    # Create normal distributions for supplier inventory ranges
    if isinstance(net_config.max_trials, dict):
        inv_dists = create_inv_dists(net_config)
    else:
        inv_dists = None

    configure_network(
        net=net_copy,
        id_firm_map=id_firm_map,
        net_config=net_config,
        edge_weight_kde=edge_weight_kde,
        product_kdes=product_kdes,
        edge_alternative_products=edge_alternative_products,
        inv_dists=inv_dists,
        product_pools=product_pools,
    )

    # 4. Create simulation config
    sim_config = SimulationConfig(
        simulation_seed=sim_config["simulation_seed"],
        disruption_seeds=sim_config["disruption_seeds"],
        strategy=sim_config["strategy"],
        dist_disruption=sim_config["dist_disruption"],
        num_parallel_requests=sim_config["num_parallel_requests"],
        network_id=sim_config["network_id"],
        scenario_id=sim_config["scenario_id"],
        num_disruption_seeds=sim_config["num_disruption_seeds"],
        sample_id=sim_config["sample_id"],
        rep_id=sim_config["rep_id"],
    )

    # 5. Run the simulation
    model = simulate_zhao(
        supply_net=net_copy,
        id_firm_map=id_firm_map,
        comp_hash=comp_hash,
        config=sim_config,
        until=diameter * 2,
        log=False,  # Keep worker logs clean
        track_time_series=True,
        validate=True,
    )
    return {
        "task_id": task_id,
        "network_id": sim_config.network_id,
        "scenario_id": sim_config.scenario_id,
        "num_seeds": sim_config.num_disruption_seeds,
        "sample_id": sim_config.sample_id,
        "rep_id": sim_config.rep_id,
        "comp_time": time.time() - start_time,
        **model.stats,
    }


# Large read-only data (network, firms, comp hash, KDEs), set in main() before the
# pool forks; the workers inherit it instead of receiving it with every task
SHARED: dict = {}


def safe_worker(config: dict) -> dict:
    """Run one simulation and return an error record instead of raising.

    An exception would otherwise end the whole pool and with it the SLURM array task.
    """
    try:
        return run_single_simulation_worker(config, **SHARED)
    except Exception as exc:
        logging.exception(f"Task {config['task_id']} failed")
        return {"task_id": config["task_id"], "error": repr(exc)}


def save_stats_to_jsonl(stats_list, filename):
    """Appends a list of stats dictionaries to a JSON Lines file."""
    logging.info(f"Writing {len(stats_list)} results to {filename}...")
    try:
        with open(filename, "a") as f:
            for stats in stats_list:
                f.write(json.dumps(stats) + "\n")
        logging.info("Write successful.")
    except IOError as e:
        logging.error(f"Failed to write to {filename}: {e}")


def main():
    """Load data and manage the multiprocessing pool."""
    parser = argparse.ArgumentParser(
        description="Run multiple Zhao model simulations in parallel on one node.",
    )
    parser.add_argument(
        "--arm",
        choices=sorted(NODE_CLASSES),
        required=True,
        help="Substitution rule: extended (product-specific) or hybrid (overlap).",
    )
    parser.add_argument(
        "--config-start",
        type=int,
        default=0,
        help="Starting index for simulation configurations.",
    )
    parser.add_argument(
        "--config-end",
        type=int,
        default=None,
        help="Ending index for simulation configurations.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to number of CPU cores.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="When to save results.",
    )
    parser.add_argument(
        "--configs",
        default="1_comp_configs_rev.json",
        help="Configuration file in <base_dir>/configs (or an absolute path).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory relative to <base_dir> (or an absolute path). "
        "Defaults to results/exp1_<arm>.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory with data/, embeddings/ and configs/.",
    )
    parser.add_argument(
        "--delivered-products",
        action="store_true",
        help="Firms can also supply the products they are observed to deliver.",
    )
    args = parser.parse_args()

    # --- Configuration ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    )

    # Use the number of CPUs available to the Slurm job, or as specified
    num_workers = args.workers or (len(os.sched_getaffinity(0))
                                   if hasattr(os, "sched_getaffinity") else os.cpu_count())
    logging.info(f"Using {num_workers} parallel workers.")

    # --- 1. Load Data ONCE in the Main Process ---
    logging.info("Starting data loading (this will happen only once)...")
    start_time = time.time()
    # Adjust these paths to your HPC's filesystem
    base_dir = Path(args.base_dir)
    net, data, id_firm_map = load_marklines(
        base_dir / "data",
        node_cls=NODE_CLASSES[args.arm],
        search_cls=ImprovedSearch,
    )
    edge_weight_kde, product_kdes, edge_alternative_products = load_edge_data(
        data=data,
        edge_product_path=base_dir / "embeddings/top_edge_products.json",
    )

    # Load configs
    with open(base_dir / "configs" / args.configs) as f:
        configs = json.load(f)
    # Filter configs based on start and end indices
    config_start = args.config_start
    config_end = args.config_end or len(configs)
    configs = configs[config_start:config_end]

    # One substitution map per call, since the competition hash is shared by all runs
    sigma_maps = {c["network_config"].get("sigma_map", DEFAULT_SIGMA_MAP) for c in configs}
    if len(sigma_maps) != 1:
        raise SystemExit(f"Configs mix substitution maps: {sorted(sigma_maps)}")
    sigma_map = sigma_maps.pop()
    logging.info(f"Substitution map: {sigma_map}")
    comp_hash = create_comp_hash_new(
        firms=id_firm_map.values(),
        path=base_dir / "embeddings" / sigma_map,
        delivered_products=(
            get_delivered_products(net) if args.delivered_products else None
        ),
    )

    # The option changes the model, so it must match what the configs were made for
    wanted = {c["network_config"].get("delivered_products", False) for c in configs}
    if wanted != {args.delivered_products}:
        msg = (f"--delivered-products={args.delivered_products} does not match the "
               f"configs (delivered_products: {wanted})")
        raise SystemExit(msg)

    # Pool sizes are only needed for the weighted imputation (sensitivity runs)
    product_pools = None
    if any(c["network_config"].get("product_weight_exponent") for c in configs):
        with open(base_dir / "embeddings/product_pools.json") as f:
            product_pools = json.load(f)

    # Delete data object to save memory before forking worker processes
    del data

    logging.info(f"Data loading finished in {time.time() - start_time:.2f} seconds.")

    # --- 2. Set up the Multiprocessing Pool ---
    # The workers inherit SHARED when the pool forks, so only the configs are
    # pickled per task (arguments of a partial would be pickled for every task).
    SHARED.update(
        base_net=net,
        id_firm_map=id_firm_map,
        comp_hash=comp_hash,
        edge_weight_kde=edge_weight_kde,
        product_kdes=product_kdes,
        edge_alternative_products=edge_alternative_products,
        product_pools=product_pools,
    )
    worker_func = safe_worker

    # --- 3. Run Simulations in Parallel ---
    logging.info(f"Starting pool of {num_workers} workers...")
    start_time = time.time()

    output_dir = base_dir / (args.output_dir or f"results/exp1_{args.arm}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_buffer = []

    # This variable will track the absolute start index of the current chunk
    chunk_start_index = config_start

    # A fresh worker per run: on nodes with strict overcommit, memory a worker keeps
    # after a large run would otherwise stay committed for the rest of the task
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        # Use imap_unordered to get results as they complete
        total_items_in_run = len(configs)
        results_iterator = pool.imap_unordered(worker_func, configs)

        # Process results one by one
        for i, stats in enumerate(results_iterator):
            # Failed runs go to a separate file so that stats files hold results only
            if "error" in stats:
                save_stats_to_jsonl([stats], output_dir / f"errors_{config_start}.jsonl")
            else:
                stats_buffer.append(stats)

            # Every chunk_size results, or if it's the very last item
            is_last_item = (i + 1) == total_items_in_run
            if len(stats_buffer) >= args.chunk_size or (is_last_item and stats_buffer):
                # The current chunk ends at the absolute index of the current item
                chunk_end_index = config_start + i
                task_range = f"{chunk_start_index}-{chunk_end_index}"

                # Ensure a unique filename if the start and end are the same
                if chunk_start_index == chunk_end_index:
                    task_range = f"{chunk_start_index}"

                output_filename = output_dir / f"stats_{task_range}.jsonl"

                save_stats_to_jsonl(stats_buffer, output_filename)
                stats_buffer.clear()  # Clear the buffer to free memory

                # The next chunk starts at the next index
                chunk_start_index = chunk_end_index + 1

    logging.info(f"All simulations finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    # Set start method to 'fork' to leverage copy-on-write memory sharing
    # This is the default on Linux but it's good to be explicit.
    multiprocessing.set_start_method("fork", force=True)
    main()
