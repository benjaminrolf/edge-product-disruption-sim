# edge-product-disruption-sim

Code for the article

> B. Rolf, H. Inoue, S. Lang. *Agent-Based Disruption Simulation on an Edge-Level
> Product-Annotated Supply Network of the Automotive Industry.* Submitted to Applied
> Network Science.

The package `graphsim` implements the agent-based disruption model of Zhao et al. (2019)
and the extensions used in the article. Supply links carry the products that flow along
them. Customers can substitute a failed supplier only with firms that offer a product
similar to the lost one, where similarity is the cosine similarity of product-name
embeddings. Capacities and inventories are calibrated to industry data. The repository
also contains the configuration generators, the runners and SLURM scripts of the three
experiments (119,000 runs), and the scripts behind every figure and table of the article.

The MarkLines data are licensed and are **not** part of this repository (see [Data](#data)).

## Contents

```
src/graphsim/            simulation package
  zhao/                  model, firm agents and substitution search
    node.py              baseline: portfolio-overlap rule of Zhao et al. (2019)
    improved_node.py     extended model: product-specific rule, calibrated capacities
    overlap_node.py      hybrid: extended model with the portfolio-overlap rule
  networks/marklines.py  builds the supply network from the MarkLines tables
  paths.py               input and output locations (environment variables)
experiments/             configuration generators and runners of Experiments 1-3
  slurm/                 SLURM array scripts used for the production runs
analysis/                network statistics, summaries, figures and tables
seeds/                   random seeds and Experiment-2 seed sets of the article
geo/tohoku_intensity.json  JMA seismic intensity per prefecture (2011 Tohoku Earthquake)
assets/                  Natural Earth country borders (public domain) for the map
```

## Installation

The runs of the article used Python 3.13. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra analysis      # simulation and analysis dependencies, pinned in uv.lock
```

The simulation alone needs only `uv sync`. `analysis/fig1_network_overview.py` also needs
the `sfdp` layout program of [Graphviz](https://graphviz.org/); sfdp 15.1.1 crashes with a
segmentation fault on the full network, so use another version.

## Data

All locations default to folders below the repository root and can be changed with
environment variables (`src/graphsim/paths.py`):

| Variable | Default | Content |
|---|---|---|
| `GRAPHSIM_DATA_DIR` | `data/` | MarkLines tables (CSV), not included |
| `GRAPHSIM_EMBEDDINGS_DIR` | `embeddings/` | product-name embeddings, not included; derived substitution maps |
| `GRAPHSIM_GEO_DIR` | `geo/` | JMA intensities (included), geocoded Japanese firms (generated) |
| `GRAPHSIM_SEEDS_DIR` | `seeds/` | random seeds (included) |
| `GRAPHSIM_CONFIGS_DIR` | `configs/` | experiment configurations (generated) |
| `GRAPHSIM_RUNS_DIR` | `results/` | simulation output, one folder per experiment |
| `GRAPHSIM_ANALYSIS_DIR` | `outputs/` | derived statistics and analysis results |
| `GRAPHSIM_FIGURES_DIR` | `outputs/figures/` | figures and the statistics behind them |

The runners in `experiments/` do not read these variables. They expect `data/`,
`embeddings/`, `configs/` and `results/` below `--base-dir` (default: the working
directory).

**MarkLines tables.** The network is built from the MarkLines Automotive Parts Suppliers
database ([marklines.com](https://www.marklines.com)), which is available under licence
from MarkLines Co., Ltd. The tables were scraped from the platform and consolidated into
firm groups (Appendix A of the article). The code reads the following files from
`data/`:

| File | Columns used |
|---|---|
| `Firms.csv` | `firm_id`, `nation_id`, `is_oem`, `latitude`, `longitude`; the notebook also uses `group_id`, `capital`, `sales`, `employees` |
| `SupplyRelations.csv` | `relation_id`, `source_firm_id`, `target_firm_id` |
| `SupplyRelationDetails.csv` | `relation_id`, `product_name`, `model_name`, `model_year` |
| `Products.csv` | `product_id`, `product_name`, `group_id`, `group_name`, `family_id`, `family_name` |
| `ProductionCapabilities.csv` | `firm_id`, `product_id` |
| `SpecialProducts.csv` | `special_product_id`, `special_product_name` |
| `SpecialProductionCapabilities.csv` | `firm_id`, `special_product_id` |
| `GeoNations.csv` | `nation_id`, `nation_name` |
| `FirmNames.csv` | `firm_id`, `firm_name`, `name_type` (only for labels in Fig. 1) |

**Embeddings.** `embeddings/product_embeddings.jsonl` (portfolio products: standard and
special products) and `embeddings/edge_product_embeddings.jsonl` (product names on
supply links) hold one OpenAI Batch API response per line, with the product name as
`custom_id` and the `text-embedding-3-large` vector (3,072 dimensions) under
`response.body.data[0].embedding`.

**Seeds.** `seeds/` holds the network, replication and sample seeds of all experiments
and the 100 disruption seed sets of Experiment 2, which were drawn without a fixed
random seed. `experiments/create_random_seeds.py` would draw new seeds, so do not run
it to reproduce the article.

## Reproducing the article

Run all commands from the repository root.

```bash
mkdir -p configs results outputs/figures logs
```

**1. Substitution maps and derived inputs**

```bash
uv run python analysis/compute_similar_products.py            # embeddings/alternative_products.json, top_edge_products.json
uv run python analysis/compute_similar_products_tau_sweep.py  # alternative_products_tau<tau>.json
uv run python analysis/compute_product_pools.py               # embeddings/product_pools.json
uv run python analysis/geolocating.py                         # geo/japanese_firms_geolocation.json
```

`compute_similar_products.py` compares every product with all others one at a time and
takes several hours on one core. `geolocating.py` downloads the prefecture borders from
[dataofjapan/land](https://github.com/dataofjapan/land).

**2. Configurations**

```bash
uv run python experiments/0_config.py           # Experiment 2
uv run python experiments/1_config.py           # Experiment 1, extended and hybrid arm
uv run python experiments/1_config_baseline.py  # Experiment 1, baseline arm
uv run python experiments/2_config.py           # Experiment 3
uv run python experiments/revision_configs.py   # final *_rev.json files read by the runners
```

**3. Simulation runs**

Each runner simulates a range of configurations in parallel and writes `stats_*.jsonl`
files. Its main options are `--config-start`, `--config-end`, `--workers`,
`--configs` and `--output-dir`.

| Experiment | Command | Runs |
|---|---|---|
| 1, baseline arm | `experiments/1_comparison_baseline.py` | 8,000 |
| 1, extended arm | `experiments/1_comparison_rev.py --arm extended --delivered-products` | 8,000 |
| 1, hybrid arm | `experiments/1_comparison_rev.py --arm hybrid --delivered-products` | 8,000 |
| 1, imputation sensitivity | as above with `--configs 1_comp_configs_rev_gamma<g>.json --output-dir results/exp1_<arm>_gamma<g>`, g in {-2, -1, +1} | 12,000 |
| 1, threshold sweep | extended arm with `--configs 1_comp_configs_rev_tau<t>.json --output-dir results/exp1_extended_tau<t>`, t in {0.65, 0.69, 0.77, 0.81, 0.85} | 10,000 |
| 2 | `experiments/0_monte_carlo.py --delivered-products` | 50,000 |
| 2, imputation sensitivity | `experiments/0_monte_carlo.py --delivered-products --configs 0_monte_carlo_configs_rev_gamma-1.json --output-dir results/exp2_gamma-1` | 500 |
| 3 | `experiments/2_geje.py --delivered-products` | 22,500 |

Run each command with `uv run python`. The full set took several thousand core-hours.
`experiments/slurm/submit_exp{1,2,3}_rev.sh` submit exactly these runs as SLURM
arrays; set `GRAPHSIM_BASE` to the repository directory on the cluster (and
`GRAPHSIM_VENV` if the environment is not `.venv`). The runs are deterministic given
the seeds in the configurations; `experiments/slurm/revision_test.sh` checks that
repeated runs give identical output.

**4. Network statistics (Sections 3 and Appendices A, B)**

| Script | Output in the article |
|---|---|
| `analysis/network_analysis.ipynb` | network statistics of Section 3, Table A1, Appendix B |
| `analysis/compute_lcc_diameter_igraph.py` | diameter (Section 3.3, Table A1) |
| `analysis/clustering_null_models.py` | clustering against null models (Appendix B.2) |
| `analysis/export_louvain_partition.py` | Louvain partition used by the scripts below |
| `analysis/louvain_intercommunity_stats.py` | links between the largest communities (Section 6.1) |
| `analysis/fig1_network_overview.py` | Fig. 1 |
| `analysis/fig1_model_schematic.py` | Fig. 2 |
| `analysis/compute_edge_centrality.py` | edge centralities read by the notebook |

**5. Validation of the imputation and the substitution rule (Section 4, Appendix C)**

| Script | Output in the article |
|---|---|
| `analysis/imputation_holdout.py [--delivered]` | Table C1 |
| `analysis/substitutability_checks.py` | Table C2 |

**6. Results (Section 5, Appendix D)**

| Script | Output in the article |
|---|---|
| `analysis/exp1_rev_summary.py --baseline results/exp1_baseline --hybrid results/exp1_hybrid --extended results/exp1_extended` | Table 2 |
| `analysis/exp1_sensitivity_summary.py --results results` | Table 3 |
| `analysis/fig_candidate_pools_edges.py --recompute` | Fig. 3 |
| `analysis/regenerate_paper_figures.py` | Figs. 4, 6 and B1 (needs the output of `geje_rev_summary.py`) |
| `analysis/exp2_rev_numbers.py --runs results/exp2 --gamma results/exp2_gamma-1 --out outputs/exp2_rev_numbers.json` | numbers of Section 5.2.1 |
| `analysis/capacity_dynamics.py --exp2 results/exp2 --exp3 results/exp3` | Fig. 5, Table D2 |
| `analysis/critical_firm_mechanism.py --runs results/exp2 --delivered-products` | Table 4 (Experiment 2) |
| `analysis/critical_firm_transfer.py --exp2-cache outputs/critical_firm_mechanism_exp2.cache.pkl --runs results/exp3 --delivered-products` | Table 4 (Experiment 3), Section 5.3 |
| `analysis/geje_rev_summary.py --runs results/exp3 --exp2 results/exp2 --out-dir outputs` | Table 5 and the numbers of Section 5.3 |
| `analysis/compute_node_betweenness.py`, then `analysis/yan_nexus_test.py` and `analysis/cascade_attribution.py` | Table D1 |
| `analysis/mc_propagation_depth.py`, `analysis/cascade_depth_degree_stats.py`, `analysis/cascade_graph_stats.py` | propagation depth (Appendix D.1) |
| `analysis/extract_cascade_example.py`, `analysis/extract_tohoku_cascade_example.py`, then `analysis/fig_cascade_graph.py` | Figs. D1 and D2 |

The figure files keep the names of an earlier figure numbering; the docstring of each
script names the figure in the article.

## Citation

Please cite the article (see `CITATION.cff`).

## Licence

MIT, see `LICENSE`. The licence covers the code only. It grants no rights to the
MarkLines data.
