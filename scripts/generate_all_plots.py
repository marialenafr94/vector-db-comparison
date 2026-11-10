#!/usr/bin/env python3

from pathlib import Path
import textwrap
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'results' / 'real'
OUT = DATA / 'plots_annotated'
OUT.mkdir(parents=True, exist_ok=True)
DBS = ['pgvector', 'vectorchord']
LABELS = {'pgvector': 'pgvector HNSW', 'vectorchord': 'VectorChord VChordRQ'}
COLORS = {'pgvector': '#3166a3', 'vectorchord': '#d17b35'}
DS = {'sift1m': 'SIFT1M (128D)', 'vibe': 'Yahoo MiniLM (384D)'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def load(relative, required):
    path = DATA / relative
    if not path.is_file():
        raise FileNotFoundError(f'Missing input CSV: {path}')
    df = pd.read_csv(path)
    absent = set(required) - set(df.columns)
    if absent:
        raise ValueError(f'{path}: missing columns {sorted(absent)}')
    if 'status' in df:
        df = df.loc[df['status'].eq('ok')].copy()
    if 'errors' in df:
        bad = int(df['errors'].fillna(0).gt(0).sum())
        if bad:
            warnings.warn(f'{path.name}: excluding {bad} rows with errors')
        df = df.loc[df['errors'].fillna(0).eq(0)].copy()
    if df.empty:
        raise ValueError(f'{path}: no usable rows')
    return df


def repetitions(df, group_cols):
    counts = df.groupby(group_cols, dropna=False).size()
    if counts.empty:
        return 'no measurements'
    low, high = int(counts.min()), int(counts.max())
    return f'{low} repetitions per plotted configuration' if low == high else f'{low}–{high} repetitions per plotted configuration'


def finish(fig, axes, name, title, caption):
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in np.ravel(axes):
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=.975)
    # Caption in a separate lower band; dynamic figure height leaves enough room.
    lines = textwrap.wrap(caption, width=112, break_long_words=False)
    fig.subplots_adjust(left=.10, right=.97, top=.84, bottom=.18 + .025 * max(0, len(lines) - 2), wspace=.30, hspace=.35)
    fig.text(.5, .045, '\n'.join(lines), ha='center', va='bottom', fontsize=8.5)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=250, facecolor='white')
    plt.close(fig)
    print(f'Created {name}.png / .pdf')


def grouped_bars(df, group_col, value_col, name, title, ylabel, caption, *, order=None):
    summary = df.groupby([group_col, 'db'])[value_col].agg(['mean', 'std', 'count']).reset_index()
    if order is None:
        order = sorted(summary[group_col].unique())
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    width = .36
    for i, db in enumerate(DBS):
        subset = summary[summary.db.eq(db)].set_index(group_col).reindex(order)
        ax.bar(x + (i - .5)*width, subset['mean'], width, label=LABELS[db], color=COLORS[db],
               yerr=subset['std'].fillna(0), capsize=3)
    ax.set_xticks(x, [DS.get(str(v), str(v)) for v in order])
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    finish(fig, ax, name, title, caption)


def index_and_loading():
    idx = load('index_build/index_build_8runs.csv', ['dataset','db','build_s','index_mib','target_rows'])
    for metric, name, title, ylabel in [
        ('build_s','01_index_build_time','Index construction time','Seconds'),
        ('index_mib','02_index_size','Index storage size','MiB')]:
        grouped_bars(idx, 'dataset', metric, name, title, ylabel,
            f'Mean {ylabel.lower()} with ±1 standard deviation across {repetitions(idx, ["dataset", "db"])}. '
            f'SIFT1M and Yahoo MiniLM; target vectors by dataset: '
            + ', '.join(f'{DS.get(str(k), k)} {int(v):,}' for k,v in idx.groupby('dataset')['target_rows'].first().items())
            + ('. Index build time excludes data COPY time.' if metric == 'build_s' else '. Index size excludes base-table size.'))

    df = load('loading/loading_8runs.csv', ['dataset','db','target_rows','elapsed_s'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    for ax, (dataset, part) in zip(axes, df.groupby('dataset', sort=True)):
        for db in DBS:
            p = part.loc[part.db.eq(db)].groupby('target_rows')['elapsed_s'].agg(['mean','std']).sort_index()
            ax.errorbar(p.index, p['mean'], yerr=p['std'].fillna(0), marker='o', capsize=3,
                        color=COLORS[db], label=LABELS[db])
        ax.set_title(DS.get(dataset, dataset))
        ax.set_xlabel('Vectors loaded')
        ax.set_ylabel('COPY loading time (seconds)')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    axes[0].legend(fontsize=8)
    finish(fig, axes, '03_loading_time', 'Data loading time versus dataset size',
           f'Mean elapsed loading time with ±1 standard deviation; {repetitions(df, ["dataset", "db", "target_rows"])}. '
           'The CSV elapsed_s metric measures the loading phase, not index construction.')


def recall_latency():
    frames=[]
    for ds, filename in [('sift1m','sift1m_recall_8runs.csv'),('vibe','vibe_yahoo_recall_8runs.csv')]:
        d=load(filename,['database','parameter','value','recall_at_10','p50_ms','repeat'])
        d['dataset_id']=ds
        frames.append(d)
    df=pd.concat(frames, ignore_index=True)
    fig, axes=plt.subplots(1,2,figsize=(12,6.6))
    for ax, ds in zip(axes, ['sift1m','vibe']):
        part=df[df.dataset_id.eq(ds)]
        for db in DBS:
            p=part[part.database.eq(db)].groupby(['parameter','value']).agg(
                recall=('recall_at_10','mean'),latency=('p50_ms','mean')).sort_values('recall')
            ax.plot(p.recall,p.latency,marker='o',label=LABELS[db],color=COLORS[db])
            for (param,value), row in p.iterrows():
                ax.annotate(f'{value:g}',(row.recall,row.latency),xytext=(3,4),textcoords='offset points',fontsize=7)
        ax.set_title(DS[ds]);ax.set_xlabel('Mean Recall@10');ax.set_ylabel('Mean reported p50 latency (ms)')
    axes[0].legend(fontsize=8)
    finish(fig,axes,'04_recall_vs_latency','Recall–latency trade-off',
        f'Each point is a search setting averaged over {repetitions(df,["dataset_id","database","parameter","value"])}. '
        'Point labels are setting values: pgvector hnsw.ef_search versus VectorChord vchordrq.epsilon (different parameters, not directly comparable numerically). '
        'Latency is the mean of per-run p50 values; not a pooled query percentile.')


def qps_plots():
    frames=[]
    for ds,filename in [('sift1m','sift1m_qps_8runs.csv'),('vibe','vibe_yahoo_qps_final.csv')]:
        d=load(filename,['database','parameter','value','clients','qps','p50_ms','repeat'])
        d['dataset_id']=ds;frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    for clients in sorted(df.clients.unique()):
        fig,axes=plt.subplots(1,2,figsize=(12,6.6))
        for ax,ds in zip(axes,['sift1m','vibe']):
            part=df[df.dataset_id.eq(ds)&df.clients.eq(clients)]
            for db in DBS:
                p=part[part.database.eq(db)].groupby(['parameter','value']).agg(
                    qps=('qps','mean'),recall_unavailable=('qps','count'),latency=('p50_ms','mean')).sort_values('latency')
                ax.plot(p.latency,p.qps,marker='o',label=LABELS[db],color=COLORS[db])
                for (_,value),row in p.iterrows():
                    ax.annotate(f'{value:g}',(row.latency,row.qps),xytext=(3,3),textcoords='offset points',fontsize=7)
            ax.set_title(DS[ds]);ax.set_xlabel('Mean reported p50 latency (ms)');ax.set_ylabel('Mean throughput (QPS)')
        axes[0].legend(fontsize=8)
        finish(fig,axes,f'05_qps_vs_latency_{int(clients)}clients',f'Throughput versus latency — {int(clients)} client(s)',
            f'Mean QPS versus mean per-run p50 latency, by search setting; {repetitions(df[df.clients.eq(clients)],["dataset_id","database","parameter","value","clients"])}. '
            'Point labels indicate ef_search for pgvector and epsilon for VectorChord. These throughput CSVs do not contain Recall@10; no recall–QPS pairing is inferred.')
    # Concurrency chart at each system's individual settings, without conflating settings
    fig,axes=plt.subplots(1,2,figsize=(12,6.6))
    for ax,ds in zip(axes,['sift1m','vibe']):
        part=df[df.dataset_id.eq(ds)]
        for db in DBS:
            sub=part[part.database.eq(db)]
            for (param,value),grp in sub.groupby(['parameter','value']):
                s=grp.groupby('clients')['qps'].mean().sort_index()
                ax.plot(s.index,s.values,marker='o',alpha=.65,color=COLORS[db],
                        label=f'{LABELS[db]} ({value:g})')
        ax.set_title(DS[ds]);ax.set_xlabel('Concurrent clients');ax.set_ylabel('Mean QPS')
        ax.set_xticks(sorted(part.clients.unique()))
    axes[0].legend(fontsize=7,ncol=2)
    finish(fig,axes,'06_qps_vs_concurrency','Throughput versus concurrent clients',
        f'Mean QPS across {repetitions(df,["dataset_id","database","parameter","value","clients"])}. '
        'Legend numbers are per-system tuning values (pgvector ef_search; VectorChord epsilon). Each line keeps its own setting fixed across client counts.')


def filtered():
    frames=[]
    for ds,filename in [('sift1m','sift1m_filtered_8runs.csv'),('vibe','vibe_yahoo_filtered_8runs.csv')]:
        d=load(filename,['database','selectivity_percent','scan_mode','search_value','recall_at_10','p50_ms'])
        d['dataset_id']=ds;frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    for metric,name,title,ylabel in [('recall_at_10','07_filtered_recall','Filtered-query Recall@10','Mean Recall@10'),
                                      ('p50_ms','08_filtered_latency','Filtered-query median latency','Mean reported p50 latency (ms)')]:
        fig,axes=plt.subplots(1,2,figsize=(13,7))
        for ax,ds in zip(axes,['sift1m','vibe']):
            part=df[df.dataset_id.eq(ds)]
            for (db,mode,value),grp in part.groupby(['database','scan_mode','search_value']):
                p=grp.groupby('selectivity_percent')[metric].mean().sort_index()
                ax.plot(p.index,p.values,marker='o',color=COLORS.get(db,'gray'),
                        linestyle='--' if db=='vectorchord' else '-',
                        alpha=.75,label=f'{LABELS.get(db,db)} / {mode} / {value:g}')
            ax.set_title(DS[ds]);ax.set_xlabel('Filter selectivity (%)');ax.set_ylabel(ylabel)
            ax.set_xticks(sorted(part.selectivity_percent.unique()))
        axes[0].legend(fontsize=6.5,ncol=2,loc='best')
        finish(fig,axes,name,title,
            f'Mean of per-run {"Recall@10" if metric=="recall_at_10" else "p50 latency"} at each filter selectivity and scan/search configuration; '
            f'{repetitions(df,["dataset_id","database","selectivity_percent","scan_mode","search_value"])}. '
            'Legend identifies the database, iterative scan mode, and search parameter value; parameter meanings differ by system.')

    frames=[]
    for ds,filename in [('sift1m','sift1m_filtered_qps_8runs.csv'),('vibe','vibe_yahoo_filtered_qps_8runs.csv')]:
        d=load(filename,['database','selectivity_percent','scan_mode','search_value','clients','qps'])
        d['dataset_id']=ds;frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    for clients in sorted(df.clients.unique()):
        fig,axes=plt.subplots(1,2,figsize=(13,7))
        for ax,ds in zip(axes,['sift1m','vibe']):
            part=df[df.dataset_id.eq(ds)&df.clients.eq(clients)]
            for (db,mode,value),grp in part.groupby(['database','scan_mode','search_value']):
                p=grp.groupby('selectivity_percent')['qps'].mean().sort_index()
                ax.plot(p.index,p.values,marker='o',color=COLORS.get(db,'gray'),
                        linestyle='--' if db=='vectorchord' else '-',alpha=.75,
                        label=f'{LABELS.get(db,db)} / {mode} / {value:g}')
            ax.set_title(DS[ds]);ax.set_xlabel('Filter selectivity (%)');ax.set_ylabel('Mean QPS')
            ax.set_xticks(sorted(part.selectivity_percent.unique()))
        axes[0].legend(fontsize=6.5,ncol=2)
        finish(fig,axes,f'09_filtered_qps_{int(clients)}clients',f'Filtered-query throughput — {int(clients)} client(s)',
            f'Mean QPS versus filter selectivity, for {int(clients)} client(s); '
            f'{repetitions(df[df.clients.eq(clients)],["dataset_id","database","selectivity_percent","scan_mode","search_value","clients"])}. '
            'Legend identifies scan mode and search setting. Comparisons are observational; search parameters differ between systems.')


def memory():
    frames=[]
    for file in ['memory_constraint/vibe_memory_unlimited.csv','memory_constraint/vibe_memory_limited_1536m.csv']:
        frames.append(load(file,['regime','database','setting','clients','qps','memory_peak_mib']))
    df=pd.concat(frames,ignore_index=True)
    regimes=sorted(df.regime.unique(),key=lambda x: 0 if x=='unlimited' else 1)
    for metric,name,title,ylabel in [('qps','10_memory_limit_qps','Throughput under container memory limits','Mean QPS'),
                                      ('memory_peak_mib','11_memory_limit_peak_memory','Peak container memory usage','Mean measured peak memory (MiB)')]:
        fig,axes=plt.subplots(1,len(regimes),figsize=(12,7),squeeze=False)
        for ax,regime in zip(axes[0],regimes):
            part=df[df.regime.eq(regime)]
            for (db,setting),grp in part.groupby(['database','setting']):
                p=grp.groupby('clients')[metric].agg(['mean','std']).sort_index()
                ax.errorbar(p.index,p['mean'],yerr=p['std'].fillna(0),marker='o',capsize=3,
                            label=f'{LABELS.get(db,db)} / {setting}',color=COLORS.get(db,'gray'),
                            linestyle='--' if 'epsilon' in setting.lower() else '-')
            ax.set_title(regime.replace('_',' '));ax.set_xlabel('Concurrent clients');ax.set_ylabel(ylabel)
            ax.set_xticks(sorted(part.clients.unique()))
        axes[0,0].legend(fontsize=7)
        finish(fig,axes,name,title,
            f'Yahoo MiniLM workload; mean and ±1 standard deviation across {repetitions(df,["regime","database","setting","clients"])}. '
            'Unlimited versus 1,536 MiB container memory limit, using the regimes recorded in the CSVs. '
            'A memory cap does not by itself establish that swapping or disk paging occurred.')


def main():
    index_and_loading()
    recall_latency()
    qps_plots()
    filtered()
    memory()
    print(f'\nAll figures saved to: {OUT}')
    print('Inputs were read only; original plots were not overwritten.')


if __name__ == '__main__':
    main()
