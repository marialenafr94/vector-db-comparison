
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "real"
OUT = ROOT / "report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"pgvector": "#2a78d6", "vectorchord": "#eb6834", "ivf": "#1baf7a"}
NAMES = {
    "pgvector": "pgvector HNSW",
    "vectorchord": "VectorChord (lists=[])",
    "ivf": "VectorChord IVF (lists=[1000])",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def series(recall_csv, qps_csv, database=None, key="value"):
    recall = pd.read_csv(DATA / recall_csv)
    qps = pd.read_csv(DATA / qps_csv)
    if database:
        recall = recall[recall.database == database]
        qps = qps[qps.database == database]
    r = recall.groupby(key).recall_at_10.mean()
    g = qps.groupby([key, "clients"]).qps
    return r, g.median(), g.quantile(0.25), g.quantile(0.75)


PANELS = [
    ("SIFT1M (128-d)", "sift1m_recall_8runs.csv", "sift1m_qps_8runs.csv", True),
    ("VIBE Yahoo MiniLM (384-d)", "vibe_yahoo_recall_8runs.csv",
     "vibe_yahoo_qps_final.csv", False),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)

for ax, (title, recall_csv, qps_csv, with_ivf) in zip(axes, PANELS):
    for target in (0.90, 0.95):
        ax.axvline(target, color="#999", ls=":", lw=0.9)

    sets = [
        ("pgvector",) + series(recall_csv, qps_csv, "pgvector"),
        ("vectorchord",) + series(recall_csv, qps_csv, "vectorchord"),
    ]
    if with_ivf:
        sets.append(("ivf",) + series(
            "vchord_ivf/sift1m_ivf_recall_8runs.csv",
            "vchord_ivf/sift1m_ivf_qps_8runs.csv",
            key="probes",
        ))

    for name, r, med, q1, q3 in sets:
        for clients, ls, marker in [(1, "-", "o"), (8, "--", "s")]:
            x = r.values
            order = np.argsort(x)
            idx = r.index
            m = med.xs(clients, level="clients").reindex(idx).values
            lo = q1.xs(clients, level="clients").reindex(idx).values
            hi = q3.xs(clients, level="clients").reindex(idx).values
            ax.errorbar(
                x[order], m[order],
                yerr=[m[order] - lo[order], hi[order] - m[order]],
                color=COLORS[name], ls=ls, marker=marker, ms=3.6, lw=1.5,
                capsize=2, elinewidth=0.8,
                label=f"{NAMES[name]}, {clients} client" + ("s" if clients > 1 else ""),
            )

    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Recall@10")

axes[0].set_ylabel("Throughput (QPS, log scale)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           fontsize=6.6, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.13, 1, 1))

for ext in ("pdf", "png"):
    path = OUT / f"recall_qps_matched.{ext}"
    fig.savefig(path, dpi=160)
    print(f"Saved {path}")
