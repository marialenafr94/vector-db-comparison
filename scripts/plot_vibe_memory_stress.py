#!/usr/bin/env python3

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results/real/memory_constraint"
OUTPUT_DIR = INPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = [
    ("unlimited", "Unlimited"),
    ("limited_1536m", "1536 MiB"),
    ("limited_1024m", "1024 MiB"),
    ("limited_768m", "768 MiB"),
]

DATABASES = ["pgvector", "vectorchord"]
CLIENTS = [1, 8]

COLORS = {
    "pgvector": "#2563EB",
    "vectorchord": "#EA580C",
}

LABELS = {
    "pgvector": "pgvector HNSW",
    "vectorchord": "VectorChord VChordRQ",
}

# Student's t critical value, 95% two-sided, df=7.
# Valid for exactly eight independent repetitions.
T_CRITICAL_95 = 2.364624

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

data = defaultdict(list)

for regime, _ in REGIMES:
    path = INPUT_DIR / f"vibe_memory_{regime}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (
                regime,
                row["database"],
                int(row["clients"]),
            )
            data[key].append(row)

for regime, _ in REGIMES:
    for database in DATABASES:
        for clients in CLIENTS:
            key = (regime, database, clients)
            rows = data[key]

            if len(rows) != 8:
                raise ValueError(
                    f"Expected 8 repetitions for {key}; "
                    f"found {len(rows)}"
                )

            if len({int(r["repeat"]) for r in rows}) != 8:
                raise ValueError(f"Duplicate/missing repeat IDs: {key}")


def values(regime, database, clients, metric):
    return [
        float(row[metric])
        for row in data[(regime, database, clients)]
    ]


def mean(regime, database, clients, metric):
    return statistics.mean(
        values(regime, database, clients, metric)
    )


def ci95(regime, database, clients, metric):
    v = values(regime, database, clients, metric)
    return (
        T_CRITICAL_95
        * statistics.stdev(v)
        / math.sqrt(len(v))
    )


def save_figure(fig, filename):
    png = OUTPUT_DIR / f"{filename}.png"
    pdf = OUTPUT_DIR / f"{filename}.pdf"

    # Do not silently overwrite previous plots.
    for path in (png, pdf):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. "
                "Move or rename it before rerunning."
            )

    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {png}")
    print(f"Created: {pdf}")


def plot_metric(metric, ylabel, title, filename, log_scale=False):
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.8), sharey=True
    )

    x = np.arange(len(REGIMES))
    regime_labels = [label for _, label in REGIMES]

    for ax, clients in zip(axes, CLIENTS):
        for database in DATABASES:
            y = [
                mean(regime, database, clients, metric)
                for regime, _ in REGIMES
            ]

            errors = [
                ci95(regime, database, clients, metric)
                for regime, _ in REGIMES
            ]

            ax.errorbar(
                x,
                y,
                yerr=errors,
                marker="o",
                markersize=6,
                linewidth=2,
                capsize=4,
                color=COLORS[database],
                label=LABELS[database],
            )

        ax.set_xticks(x, regime_labels)
        ax.set_title(f"{clients} concurrent client(s)")
        ax.set_xlabel("Container memory regime")
        ax.grid(axis="y", alpha=0.25)

        if log_scale:
            ax.set_yscale("log")

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.035),
        frameon=False,
    )

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.83,
        bottom=0.23,
        wspace=0.12,
    )

    save_figure(fig, filename)


# Figure 1: Throughput with 95% confidence intervals
plot_metric(
    metric="qps",
    ylabel="Queries per second (QPS)",
    title="VIBE: Throughput under memory constraints",
    filename="vibe_memory_qps_ci95",
)

# Figure 2: p95 latency
plot_metric(
    metric="p95_ms",
    ylabel="Mean per-run p95 latency (ms)",
    title="VIBE: p95 query latency under memory constraints",
    filename="vibe_memory_p95",
    log_scale=True,
)

# Figure 3: p99 latency
plot_metric(
    metric="p99_ms",
    ylabel="Mean per-run p99 latency (ms)",
    title="VIBE: p99 query latency under memory constraints",
    filename="vibe_memory_p99",
    log_scale=True,
)

# Figure 4: Throughput change relative to 1024 MiB
fig, ax = plt.subplots(figsize=(9, 5))

categories = [
    "pgvector\n1 client",
    "pgvector\n8 clients",
    "VectorChord\n1 client",
    "VectorChord\n8 clients",
]

changes = []
bar_colors = []

for database in DATABASES:
    for clients in CLIENTS:
        baseline = mean(
            "limited_1024m", database, clients, "qps"
        )
        constrained = mean(
            "limited_768m", database, clients, "qps"
        )

        changes.append(
            100 * (constrained / baseline - 1)
        )
        bar_colors.append(COLORS[database])

bars = ax.bar(
    categories,
    changes,
    color=bar_colors,
    width=0.62,
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("QPS change (%)")
ax.set_title(
    "VIBE: Throughput change from 1024 MiB to 768 MiB",
    fontweight="bold",
)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

for bar, change in zip(bars, changes):
    ax.annotate(
        f"{change:+.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, change),
        xytext=(0, 5 if change >= 0 else -5),
        textcoords="offset points",
        ha="center",
        va="bottom" if change >= 0 else "top",
        fontweight="bold",
    )

fig.tight_layout()
save_figure(fig, "vibe_memory_qps_change_1024_to_768")

print("\nAll four figures generated successfully.")
