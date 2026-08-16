"""Lightweight plotting of the tile-budget degradation curves.

Reads the per-fold metrics that ``eval_degradation`` writes
(``<results_dir>/fold_*_metrics.csv``), aggregates AUROC mean±sd across folds, and
renders the degradation curve. Imports no torch / ML stack, so regenerating a plot
(e.g. after tweaking the title or colours) is instant instead of paying the eval's
model-loading startup.

CLI:
    python -m tileselect.eval.plot --results_dir <data_root>/eval_results/<run> \
        --domain <cohort> --config configs/mycohort.yaml
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# The five strategies that make up the degradation plot, with fixed styling.
COLORS = {
    'random': '#4169E1',
    'attn': '#DC143C',
    'unet': '#800080',
    'attn_sample': '#FF8C00',
    'unet_sample': '#00CED1',
}
LABELS = {
    'random': 'Random',
    'attn': 'Top tiles Attention',
    'unet': 'Top tiles U-Net',
    'attn_sample': 'Attention weighted sampling',
    'unet_sample': 'U-Net weighted sampling',
}
STYLES = {'random': '-', 'attn': '-', 'unet': '-', 'attn_sample': '-', 'unet_sample': '-'}
MARKERS = {'random': 'o', 'attn': 'o', 'unet': 'o', 'attn_sample': 's', 'unet_sample': 'D'}
# Optional pretty names per cohort tag; anything absent falls back to the tag itself.
TASK_TITLES = {}
PLOT_ORDER = ['random', 'attn', 'unet', 'attn_sample', 'unet_sample']


def build_summary(results_dir, strategies=None, max_fold=None, band='fold', n_resample=None):
    """Aggregate ``fold_*_metrics.csv`` into a per-(strategy, pct) mean±sd frame.

    ``band`` controls what the ``auc_std`` (the shaded band) represents:
      - ``'fold'`` (default): std of the per-fold AUROC ACROSS folds — the classic
        cross-validation spread. Collapses to 0 on a single fold.
      - ``'resample'``: the RESAMPLING spread σ_within — sqrt of the MEAN over folds of
        the per-fold resampling variance (each fold's ``auc_std`` = std over the stochastic
        strategies' weighted-sampling trials). It is the within-fold term of the total
        decomposition on its own: how much the AUROC moves if the weighted sample is
        re-drawn (does NOT include the cross-fold spread). Deterministic strategies (top-k) → 0.
      - ``'total'``: law of total variance — σ²_total = σ²_between + σ²_within, i.e.
        Var_folds(per-fold mean AUROC) [cross-val] + mean_folds(per-fold resampling var)
        [sampling]. The single correct band across BOTH families: for deterministic
        strategies σ²_within=0 so it reduces to the cross-fold band; for the weighted-
        sampling strategies it combines both sources. Needs ≥2 folds for σ²_between.
    """
    agg = defaultdict(lambda: {'aucs': [], 'auprcs': [], 'stds': []})
    bcol = 'pct'
    for fp in sorted(glob.glob(os.path.join(results_dir, "fold_*_metrics.csv"))):
        try:
            fold = int(os.path.basename(fp).split('_')[1])
        except (IndexError, ValueError):
            continue
        if max_fold is not None and fold > max_fold:
            continue
        df = pd.read_csv(fp)
        # Budget axis is either a per-slide fraction ('pct') or an absolute tile count ('k').
        bcol = 'k' if 'k' in df.columns else 'pct'
        for _, row in df.iterrows():
            if strategies is not None and row['strategy'] not in strategies:
                continue
            agg[(row['strategy'], row[bcol])]['aucs'].append(row['auc'])
            agg[(row['strategy'], row[bcol])]['auprcs'].append(row['auprc'])
            rs = row['auc_std'] if ('auc_std' in row and pd.notna(row['auc_std'])) else 0.0
            agg[(row['strategy'], row[bcol])]['stds'].append(float(rs))
    rows = []
    for (strat, budget), vals in agg.items():
        if not vals['aucs']:
            continue
        if band == 'resample':
            # σ_within = sqrt of the MEAN of the per-fold resampling VARIANCES (not the mean
            # of the sds — by Jensen that would underestimate). Matches the within-fold term
            # of the 'total' band, so the two are consistent.
            auc_std = float(np.sqrt(np.mean(np.square(vals['stds'])))) if vals['stds'] else 0.0
        elif band == 'total':
            # law of total variance: between-fold var + averaged within-fold (resampling) var
            v_between = float(np.var(vals['aucs'])) if len(vals['aucs']) > 1 else 0.0
            v_within = float(np.mean(np.square(vals['stds']))) if vals['stds'] else 0.0
            # Optional bias correction: each per-fold mean carries residual sampling noise
            # v_within/R, so the naive Var_f overestimates the true between-fold variance.
            # Unbiased σ²_between = Var_f(auc_f) − v_within/R (clamped at 0).
            if n_resample and n_resample > 1 and len(vals['aucs']) > 1:
                v_between = max(0.0, v_between - v_within / float(n_resample))
            auc_std = float(np.sqrt(v_between + v_within))
        else:
            auc_std = float(np.std(vals['aucs'])) if len(vals['aucs']) > 1 else 0.0
        rows.append({'strategy': strat, bcol: budget,
                     'auc': float(np.mean(vals['aucs'])),
                     'auc_std': auc_std,
                     'auprc': float(np.mean(vals['auprcs']))})
    return pd.DataFrame(rows)


def plot_degradation(summary_df, domain, split='test', plots_dir='.',
                     proxy_tag='combined', current_fold=None, band='fold'):
    """Render the degradation curve for ``summary_df`` and save it. Returns the path.

    ``band`` only annotates the title with what the shaded region means
    ('resample' → weighted-sampling resampling spread; 'fold' → cross-fold spread).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Budget axis: 'k' = absolute tile count (full budget encoded as the 0 sentinel);
    # 'pct' = per-slide fraction (full budget = 1.0).
    bcol = 'k' if 'k' in summary_df.columns else 'pct'
    full_val = 0 if bcol == 'k' else 1.0

    # The full-budget point is always drawn. On the pct axis 1.0 sits on the log scale
    # naturally; on the k axis the 0 sentinel has no log position, so it is placed one
    # octave beyond the largest k and its tick is labelled ALL.
    real_ks = sorted(b for b in summary_df[bcol].unique() if not np.isclose(float(b), full_val))
    full_x = (max(real_ks) * 2 if real_ks else 1) if bcol == 'k' else 1.0
    xpos = lambda b: full_x if np.isclose(float(b), full_val) else float(b)

    plt.figure(figsize=(12, 7))
    for strat in PLOT_ORDER:
        sd = summary_df[summary_df['strategy'] == strat].copy()
        if sd.empty:
            continue
        sd['_x'] = sd[bcol].map(xpos)
        sd = sd.sort_values('_x', ascending=False)
        plt.plot(sd['_x'], sd['auc'], color=COLORS[strat], label=f"{LABELS[strat]} (AUROC)",
                 linestyle=STYLES[strat], marker=MARKERS[strat], linewidth=2, markersize=6)
        if sd['auc_std'].sum() > 0:
            plt.fill_between(sd['_x'], sd['auc'] - sd['auc_std'], sd['auc'] + sd['auc_std'],
                             color=COLORS[strat], alpha=0.15)

    full = summary_df[np.isclose(summary_df[bcol].astype(float), full_val)]
    if not full.empty:
        baseline = float(full['auc'].iloc[0])   # any strategy at 100% tiles = same (no selection)
        plt.axhline(baseline, color='black', linestyle='--', linewidth=2.5, zorder=10,
                    label=f"all tiles baseline ({baseline:.3f})")
        if bcol == 'k':
            # Mark that ALL is not a point on the k scale.
            plt.axvline(full_x * 0.71, color='#999999', linestyle=':', linewidth=1.2, zorder=1)

    task_name = TASK_TITLES.get(domain.lower(), domain.upper())
    title = f'CLAM Tile-Budget Degradation (AUROC) — {task_name}'
    if current_fold is not None:
        title += f' (Up to Fold {current_fold})'
    if band == 'resample':
        title += '\nband = weighted-sampling resampling spread (single fold)'
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Tiles Kept per Slide (k) - Logarithmic Scale' if bcol == 'k'
               else 'Context Kept (%) - Logarithmic Scale', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    if bcol == 'k':
        ticks = [full_x] + sorted(real_ks, reverse=True)
        labels = ['ALL'] + [f"{int(t)}" for t in sorted(real_ks, reverse=True)]
    else:
        ticks = sorted(summary_df[bcol].unique(), reverse=True)
        labels = [f"{t*100:g}%" for t in ticks]
    plt.xticks(ticks, labels, rotation=45, ha='right')
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.legend(loc='lower left', fontsize=10)

    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'degradation_curves_{domain.lower()}_{split}_{proxy_tag}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="tileselect-plot",
                                description="Re-plot degradation curves from eval_results CSVs (no torch).")
    p.add_argument('--results_dir', required=True, help="Dir with fold_*_metrics.csv, under <data_root>/eval_results/")
    p.add_argument('--domain', required=True, help="Cohort tag, used for titles and filenames")
    p.add_argument('--split', default='test')
    p.add_argument('--proxy_tag', default='combined')
    p.add_argument('--band', default='fold', choices=['fold', 'resample', 'total'],
                   help="What the shaded band represents: 'fold' = cross-fold spread (default); "
                        "'resample' = weighted-sampling resampling spread (natural for a single fold); "
                        "'total' = law of total variance √(σ²_between + σ²_within), correct across "
                        "both top-k (reduces to cross-fold) and sampling (multi-fold).")
    p.add_argument('--plots_dir', default=None, help="Output dir; else derived from --config's paths.data_root")
    p.add_argument('--config', default=None, help="YAML config, used to resolve plots_dir when --plots_dir is omitted")
    args = p.parse_args(argv)

    plots_dir = args.plots_dir
    if plots_dir is None and args.config:
        import yaml
        from tileselect import paths
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        plots_dir = paths.plots_dir(cfg)
    if plots_dir is None:
        plots_dir = '.'

    summary = build_summary(args.results_dir, band=args.band)
    if summary.empty:
        sys.exit(f"No fold_*_metrics.csv found in {args.results_dir}")
    path = plot_degradation(summary, args.domain, split=args.split,
                            plots_dir=plots_dir, proxy_tag=args.proxy_tag, band=args.band)
    bc = 'k' if 'k' in summary.columns else 'pct'
    vals = sorted(summary[bc].unique())
    if bc == 'k':
        ks = [v for v in vals if v > 0]
        rng = f"ALL + k={int(ks[-1])}..{int(ks[0])}"
    else:
        rng = f"{vals[-1]*100:g}%..{vals[0]*100:g}%"
    print(f"Plot saved to {path}  ({summary['strategy'].nunique()} strategies, {rng})")


if __name__ == "__main__":
    main()
