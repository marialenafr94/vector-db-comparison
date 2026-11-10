#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/real')
OUT = ROOT / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {'pgvector': '#2367ad', 'vectorchord': '#e07828'}
MARKERS = {'pgvector': 'o', 'vectorchord': 's'}
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 220, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

def load(filename, required):
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run this script from ~/vector-db-comparison')
    df = pd.read_csv(path)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{filename} missing columns: {sorted(missing)}. Found: {list(df.columns)}')
    return df

def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, bbox_inches='tight')
        print('Saved:', path)
    plt.close(fig)

def basic_bar(filename, value_col, ylabel, title, name):
    df = load(filename, ['database', value_col]).set_index('database')
    labels = ['pgvector', 'vectorchord']
    values = [float(df.loc[d, value_col]) for d in labels]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    bars = ax.bar(['pgvector HNSW', 'VectorChord VChordRQ'], values, color=[COLORS[d] for d in labels], width=.55)
    ax.bar_label(bars, fmt='%.1f', padding=4)
    ax.set_ylim(0, max(values) * 1.17)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    save(fig, name)

basic_bar('sift1m_index_build.csv', 'index_build_seconds', 'Build time (seconds)', 'SIFT1M ANN index construction (4 CPU / 8 GB)', '01_index_build_time')
basic_bar('sift1m_index_build.csv', 'index_mb', 'Index size (MB)', 'SIFT1M ANN index storage', '02_index_size')

recall = load('sift1m_unified_recall_latency.csv', ['database', 'parameter', 'value', 'recall_at_10', 'p50_ms'])
fig, ax = plt.subplots(figsize=(7, 4.8))
for db in ('pgvector', 'vectorchord'):
    part = recall[recall.database == db].sort_values('recall_at_10')
    ax.plot(part.recall_at_10 * 100, part.p50_ms, marker=MARKERS[db], color=COLORS[db], linewidth=2, label=db)
    for _, row in part.iterrows():
        ax.annotate(f'{row.value:g}', (row.recall_at_10 * 100, row.p50_ms), xytext=(4, 4), textcoords='offset points', fontsize=8)
ax.set_xlabel('Recall@10 (%)')
ax.set_ylabel('Median latency p50 (ms)')
ax.set_title('SIFT1M accuracy–latency tradeoff\n(point labels: ef_search or epsilon)')
ax.legend()
ax.grid(alpha=.2)
save(fig, '03_recall_vs_latency')

qps = load('sift1m_qps_final.csv', ['database', 'parameter', 'value', 'clients', 'repeat', 'qps'])
# Join on the exact parameter identity; recall and throughput came from different runs.
qps_summary = qps.groupby(['database', 'parameter', 'value', 'clients'], as_index=False).agg(mean_qps=('qps', 'mean'), sd_qps=('qps', 'std'))
joined = qps_summary.merge(recall[['database', 'parameter', 'value', 'recall_at_10']], on=['database', 'parameter', 'value'], how='left', validate='many_to_one')
if joined.recall_at_10.isna().any():
    raise ValueError('Some QPS settings have no matching recall measurement. Check parameter/value types.')
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
for ax, clients in zip(axes, (1, 8)):
    for db in ('pgvector', 'vectorchord'):
        part = joined[(joined.database == db) & (joined.clients == clients)].sort_values('recall_at_10')
        ax.errorbar(part.recall_at_10 * 100, part.mean_qps, yerr=part.sd_qps.fillna(0), marker=MARKERS[db], color=COLORS[db], linewidth=2, capsize=3, label=db)
    ax.set_title(f'{clients} concurrent client' + ('s' if clients > 1 else ''))
    ax.set_xlabel('Recall@10 (%)')
    ax.grid(alpha=.2)
    ax.legend()
axes[0].set_ylabel('Mean throughput (QPS), error bars ±1 SD')
fig.suptitle('SIFT1M recall–throughput tradeoff (3 × 30 s per setting)')
save(fig, '04_recall_vs_qps')

filtered = load('sift1m_filtered_results.csv', ['database', 'selectivity_percent', 'scan_mode', 'search_value', 'recall_at_10', 'p50_ms', 'incomplete_queries'])
def series_name(row):
    if row.database == 'pgvector':
        return f'HNSW {row.scan_mode}'
    return f'VChordRQ epsilon={row.search_value:g}'
filtered['series'] = filtered.apply(series_name, axis=1)
series = ['HNSW off', 'HNSW strict_order', 'HNSW relaxed_order', 'VChordRQ epsilon=0.5', 'VChordRQ epsilon=0.7']
palette = ['#8ab3db', '#2367ad', '#123b66', '#f4b17a', '#e07828']

def filtered_plot(metric, ylabel, title, filename, percent=False):
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    selectivities = [50, 10, 1]
    x = np.arange(3)
    width = .16
    for j, (label, color) in enumerate(zip(series, palette)):
        part = filtered[filtered.series == label].set_index('selectivity_percent')
        vals = [float(part.loc[s, metric]) * (100 if percent else 1) for s in selectivities]
        ax.bar(x + (j - 2) * width, vals, width=width, label=label, color=color)
    ax.set_xticks(x, ['50%', '10%', '1%'])
    ax.set_xlabel('Filter selectivity (eligible share of vectors)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.set_ylim(0, 108)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=9, loc='upper left', bbox_to_anchor=(0, -.18))
    save(fig, filename)

filtered_plot('recall_at_10', 'Recall@10 (%)', 'SIFT1M filtered recall (100 queries; single run)', '05_filtered_recall', percent=True)
filtered_plot('p50_ms', 'Median latency p50 (ms)', 'SIFT1M filtered query latency (single run)', '06_filtered_latency')
print('\nDone: 6 figures × PNG/PDF in', OUT)
