"""Inter-community connectivity of the five largest Louvain blocs.

For the canonical Louvain partition (seed 42, see export_louvain_partition.py),
counts supply edges within and between the five largest communities and checks
which bloc is the largest external partner of each other bloc. Quantifies the
"transatlantic bloc sits structurally in the middle" claim of the discussion
and the bridge-firm hypothesis of the Tohoku case study.

Output: outputs/figures/louvain_intercommunity_stats.json
"""

import json
from collections import Counter

from graphsim.networks.marklines import build_network, parse
from graphsim import paths

PARTITION = paths.ANALYSIS_DIR / "louvain_partition_seed42.json"
OUTPUT = paths.FIGURES_DIR / "louvain_intercommunity_stats.json"

TOP = 5
NAMES = {0: "China-led", 1: "Japan-led", 2: "transatlantic",
         3: "India-led", 4: "Korea-led"}


def main() -> None:
    data = parse(str(paths.DATA_DIR))
    net = build_network(data)
    membership = {int(k): v for k, v in
                  json.loads(PARTITION.read_text())["membership"].items()}

    intra = Counter()
    inter = Counter()  # per community, counting each inter-bloc edge at both ends
    pair = Counter()
    for u, v in net.edges():
        cu, cv = membership.get(u), membership.get(v)
        if cu is None or cv is None or cu >= TOP or cv >= TOP:
            continue
        if cu == cv:
            intra[cu] += 1
        else:
            inter[cu] += 1
            inter[cv] += 1
            pair[tuple(sorted((cu, cv)))] += 1

    stats = {}
    for c in range(TOP):
        total = intra[c] + inter[c]
        stats[NAMES[c]] = {
            "intra_edges": intra[c],
            "inter_edges": inter[c],
            "inter_share": round(inter[c] / total, 4),
        }
        print(f"{NAMES[c]}: intra={intra[c]}, inter={inter[c]}, "
              f"share={inter[c] / total:.1%}")

    pair_counts = {f"{NAMES[a]} -- {NAMES[b]}": n
                   for (a, b), n in pair.most_common()}
    for k, n in pair_counts.items():
        print(f"{k}: {n}")

    # is one bloc the largest external partner of every other bloc?
    largest_partner = {}
    for c in range(TOP):
        partners = {other: pair[tuple(sorted((c, other)))]
                    for other in range(TOP) if other != c}
        best = max(partners, key=partners.get)
        largest_partner[NAMES[c]] = NAMES[best]
        print(f"largest external partner of {NAMES[c]}: {NAMES[best]} "
              f"({partners[best]} edges)")

    OUTPUT.write_text(json.dumps({
        "per_community": stats,
        "pairwise_edges": pair_counts,
        "largest_external_partner": largest_partner,
    }, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
