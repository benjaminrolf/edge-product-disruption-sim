"""Final configurations of Experiments 1-3 (the *_rev.json files that the runners read).

Derived from the files written by 0_config.py, 1_config.py, 1_config_baseline.py and
2_config.py, so that disruption seeds, network seeds and simulation seeds are
unchanged. Changes:

  all experiments  dist_cap["no_customer_capacity"] = 1: firms without recorded
                   customers (isolates and pure customers) get a free capacity of
                   one unit instead of the utilisation-based value of about 0.
  delivered        delivered_products = true for all extended and hybrid runs: firms
                   can also supply the products they are observed to deliver
                   (runners check that --delivered-products matches).
  Experiment 1     extended/hybrid arms get the calibration of the paper (Table 1):
                   tier-differentiated days-of-supply trial budgets and three
                   parallel requests. 1_comp_configs.json and
                   1_comp_configs_baseline.json share their disruption seeds (checked).
  Experiment 2     utilisation rate mu = 0.75 as in Table 1 (checked).
  Imputation       sensitivity of Experiment 1: scenarios 0 and 4
                   (top-10% degree and degree-1 seeds) with product_weight_exponent
                   gamma in SENSITIVITY_GAMMAS; task ids are kept, so every run pairs
                   with the main run (gamma = 0) of the same task. Experiment 2:
                   |S| = 100, one seed sample per network (500 runs), calibrated gamma.
  Tau sweep        of Experiment 1, extended arm: scenarios 0 and 4
                   with the substitution map of each tau in TAU_SWEEP
                   (network_config["sigma_map"]); task ids are kept. tau = 0.73 is the
                   main run: alternative_products_tau0.73.json has the same Sigma sets
                   in the same order as alternative_products.json (scores differ by
                   less than 1e-5), so it is not run again.

Outputs (next to the inputs): 1_comp_configs_rev.json, 0_monte_carlo_configs_rev.json,
2_geje_configs_rev.json, 1_comp_configs_rev_gamma<gamma>.json,
0_monte_carlo_configs_rev_gamma<gamma>.json, 1_comp_configs_rev_tau<tau>.json.
"""

import json
from graphsim import paths

RESULTS_DIR = paths.CONFIGS_DIR

NO_CUSTOMER_CAPACITY = 1
DELIVERED_PRODUCTS = True
PERIOD_LEN = 7
MAX_TRIALS = {
    "t1": {"mu": 12 / PERIOD_LEN, "sigma": 5 / PERIOD_LEN, "lb": 0, "ub": 10_000},
    "t2": {"mu": 35 / PERIOD_LEN, "sigma": 5 / PERIOD_LEN, "lb": 0, "ub": 10_000},
}
NUM_PARALLEL_REQUESTS = 3
UTILISATION_MU = 0.75
# -1 is the gamma calibrated on the annotated edges (imputation_holdout.py --delivered),
# -2 a stronger tilt towards scarce products, +1 a tilt towards substitutable ones
SENSITIVITY_GAMMAS = (-2.0, -1.0, 1.0)
SENSITIVITY_GAMMA_EXP2 = -1.0
SENSITIVITY_SCENARIOS = (0, 4)
TAU_SWEEP = (0.65, 0.69, 0.77, 0.81, 0.85)


def load(name: str) -> list[dict]:
    with open(RESULTS_DIR / name, encoding="utf-8") as fp:
        return json.load(fp)


def save(configs: list[dict], name: str) -> None:
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(configs))
    print(f"Wrote {len(configs)} configurations to {name}")


def check_calibration(net_config: dict) -> None:
    assert net_config["dist_cap"]["strategy"] == "capacity_utilization"
    assert net_config["dist_cap"]["mu"] == UTILISATION_MU
    assert net_config["max_trials"] == MAX_TRIALS


def experiment_1() -> None:
    configs = load("1_comp_configs.json")
    baseline = load("1_comp_configs_baseline.json")
    for config, base in zip(configs, baseline, strict=True):
        sim_config, net_config = config["simulation_config"], config["network_config"]
        assert (sim_config["disruption_seeds"]
                == base["simulation_config"]["disruption_seeds"])
        sim_config.pop("max_trials")
        sim_config["num_parallel_requests"] = NUM_PARALLEL_REQUESTS
        net_config["max_trials"] = MAX_TRIALS
        net_config["dist_cap"]["no_customer_capacity"] = NO_CUSTOMER_CAPACITY
        net_config["delivered_products"] = DELIVERED_PRODUCTS
        check_calibration(net_config)
    save(configs, "1_comp_configs_rev.json")


def experiment_1_sensitivity() -> None:
    configs = [c for c in load("1_comp_configs_rev.json")
               if c["simulation_config"]["scenario_id"] in SENSITIVITY_SCENARIOS]
    for gamma in SENSITIVITY_GAMMAS:
        for config in configs:
            config["network_config"]["product_weight_exponent"] = gamma
        save(configs, f"1_comp_configs_rev_gamma{gamma:+g}.json")


def experiment_1_tau_sweep() -> None:
    configs = [c for c in load("1_comp_configs_rev.json")
               if c["simulation_config"]["scenario_id"] in SENSITIVITY_SCENARIOS]
    for tau in TAU_SWEEP:
        for config in configs:
            config["network_config"]["sigma_map"] = f"alternative_products_tau{tau}.json"
        save(configs, f"1_comp_configs_rev_tau{tau}.json")


def experiment_2() -> None:
    configs = load("0_monte_carlo_configs_v2.json")
    for config in configs:
        net_config = config["network_config"]
        net_config["dist_cap"]["mu"] = UTILISATION_MU
        net_config["dist_cap"]["no_customer_capacity"] = NO_CUSTOMER_CAPACITY
        net_config["delivered_products"] = DELIVERED_PRODUCTS
        check_calibration(net_config)
        assert config["simulation_config"]["num_parallel_requests"] == 3
    save(configs, "0_monte_carlo_configs_rev.json")


def experiment_2_sensitivity() -> None:
    configs = [c for c in load("0_monte_carlo_configs_rev.json")
               if c["simulation_config"]["num_disruption_seeds"] == 100
               and c["simulation_config"]["sample_id"] == 0]
    assert len({c["simulation_config"]["network_id"] for c in configs}) == len(configs)
    for config in configs:
        config["network_config"]["product_weight_exponent"] = SENSITIVITY_GAMMA_EXP2
    save(configs, f"0_monte_carlo_configs_rev_gamma{SENSITIVITY_GAMMA_EXP2:+g}.json")


def experiment_3() -> None:
    configs = load("2_geje_configs.json")
    for config in configs:
        net_config = config["network_config"]
        net_config["dist_cap"]["no_customer_capacity"] = NO_CUSTOMER_CAPACITY
        net_config["delivered_products"] = DELIVERED_PRODUCTS
        check_calibration(net_config)
        assert config["simulation_config"]["num_parallel_requests"] == 3
    save(configs, "2_geje_configs_rev.json")


if __name__ == "__main__":
    experiment_1()
    experiment_1_sensitivity()
    experiment_1_tau_sweep()
    experiment_2()
    experiment_2_sensitivity()
    experiment_3()
