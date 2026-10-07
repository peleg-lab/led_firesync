"""B1 -- spontaneous IFI mean / Var / CV / quality factor (censored vs uncensored).

For each trial we use the *habituation* segment (FF onsets before the first LED
pulse).  We compute, both pooling across all trials and per-trial:

  mean(T), Var[T], CV = sigma/mu, mean rate r = 1/mu,
  QF = 1 / (2*pi*r^2*Var[T]).

Two filtering regimes:
  * uncensored : all inter-flash intervals in the habituation segment.
  * censored   : intervals restricted to bouts (gap <= 2 s, bout >= 3 flashes),
                 mirroring the paper's bout-segmentation rule.

Outputs:
  analyses/results/B1_ifi_stats.csv      (summary table, pooled + per-trial)
  analyses/results/B1_ifi_per_trial.csv  (one row per trial, both regimes)
  analyses/figs/B1_ifi_kde.pdf           (overlaid KDE censored vs uncensored)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data_loader import (  # noqa: E402
    bout_segments, interflash_intervals, iter_trials,
)
from lib.plot_style import apply_rc, clean_all  # noqa: E402

apply_rc()

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / "figs"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


def summarize(ifis: np.ndarray) -> dict:
    if ifis.size < 2:
        return {"n": int(ifis.size), "mean": np.nan, "var": np.nan,
                "std": np.nan, "cv": np.nan, "rate": np.nan, "qf": np.nan}
    mean, var = float(np.mean(ifis)), float(np.var(ifis, ddof=1))
    std = float(np.sqrt(var))
    cv = std / mean if mean > 0 else np.nan
    rate = 1.0 / mean
    qf = 1.0 / (2.0 * np.pi * rate * rate * var) if var > 0 else np.nan
    return {"n": int(ifis.size), "mean": mean, "var": var, "std": std,
            "cv": cv, "rate": rate, "qf": qf}


def main() -> None:
    rows = []
    pooled_unc, pooled_cen = [], []
    pooled_ent_unc, pooled_ent_cen = [], []
    for trial in iter_trials():
        hab = trial.ff_habituation
        unc = interflash_intervals(hab)
        cen_pieces = bout_segments(hab, gap_s=2.0, min_flashes=3)
        cen = np.concatenate([np.diff(p) for p in cen_pieces]) if cen_pieces else np.array([])
        pooled_unc.append(unc); pooled_cen.append(cen)

        # entrained (LED-on) segment: same two filtering regimes
        pert = trial.ff_perturbed
        ent_unc = interflash_intervals(pert)
        ent_cen_pieces = bout_segments(pert, gap_s=2.0, min_flashes=3)
        ent_cen = (np.concatenate([np.diff(p) for p in ent_cen_pieces])
                   if ent_cen_pieces else np.array([]))
        pooled_ent_unc.append(ent_unc); pooled_ent_cen.append(ent_cen)

        rows.append({
            "date": trial.date, "led_period_ms": trial.led_period_ms,
            "index": trial.index,
            "habituation_s": trial.habituation_duration,
            **{f"unc_{k}": v for k, v in summarize(unc).items()},
            **{f"cen_{k}": v for k, v in summarize(cen).items()},
            **{f"ent_unc_{k}": v for k, v in summarize(ent_unc).items()},
            **{f"ent_cen_{k}": v for k, v in summarize(ent_cen).items()},
        })

    per_trial = pd.DataFrame(rows)
    per_trial.to_csv(RESULTS / "B1_ifi_per_trial.csv", index=False)

    pooled_unc_arr = np.concatenate([x for x in pooled_unc if x.size])
    pooled_cen_arr = np.concatenate([x for x in pooled_cen if x.size])
    pooled_ent_unc_arr = np.concatenate([x for x in pooled_ent_unc if x.size])
    pooled_ent_cen_arr = np.concatenate([x for x in pooled_ent_cen if x.size])

    summary = pd.DataFrame([
        {"regime": "spontaneous, uncensored (pooled)", **summarize(pooled_unc_arr)},
        {"regime": "spontaneous, censored (pooled)",   **summarize(pooled_cen_arr)},
        {"regime": "entrained, uncensored (pooled)",   **summarize(pooled_ent_unc_arr)},
        {"regime": "entrained, censored (pooled)",     **summarize(pooled_ent_cen_arr)},
        {"regime": "spontaneous, uncensored (trial mean)",
         **{k: float(np.nanmean(per_trial[f"unc_{k}"])) for k in
            ["n", "mean", "var", "std", "cv", "rate", "qf"]}},
        {"regime": "spontaneous, censored (trial mean)",
         **{k: float(np.nanmean(per_trial[f"cen_{k}"])) for k in
            ["n", "mean", "var", "std", "cv", "rate", "qf"]}},
        {"regime": "entrained, uncensored (trial mean)",
         **{k: float(np.nanmean(per_trial[f"ent_unc_{k}"])) for k in
            ["n", "mean", "var", "std", "cv", "rate", "qf"]}},
        {"regime": "entrained, censored (trial mean)",
         **{k: float(np.nanmean(per_trial[f"ent_cen_{k}"])) for k in
            ["n", "mean", "var", "std", "cv", "rate", "qf"]}},
    ])
    summary.to_csv(RESULTS / "B1_ifi_stats.csv", index=False)
    print(summary.to_string(index=False))

    # ---- figure: stacked histograms, matching Fig. 2 style
    # (linear x-axis, 0.033 s bins, gray top row = uncensored, blue bottom
    # row = bout-censored)
    X_MAX = 5.0           # seconds (sufficient to show within-bout structure;
                          # uncensored tail extends beyond and is noted in caption)
    BIN_WIDTH = 0.033     # seconds, matching Fig. 2's bin width
    bin_edges = np.arange(0.0, X_MAX + BIN_WIDTH, BIN_WIDTH)

    stats_unc = summarize(pooled_unc_arr)
    stats_cen = summarize(pooled_cen_arr)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)

    unc_color = "#7f7f7f"
    cen_color = "#1f77b4"

    for ax, arr, color, label, stats in (
        (axes[0], pooled_unc_arr, unc_color, "uncensored", stats_unc),
        (axes[1], pooled_cen_arr, cen_color, "bout-censored", stats_cen),
    ):
        ax.hist(arr, bins=bin_edges, density=True, color=color,
                edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.axvline(np.median(arr), ls="--", color="k", alpha=0.6, lw=0.9)
        ax.text(0.98, 0.92,
                f"{label}\n"
                rf"$\overline{{T}}_{{FF}} = {stats['mean']:.2f}$ s, "
                rf"CV$ = {stats['cv']:.2f}$, QF$ = {stats['qf']:.3f}$",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="none", alpha=0.85))
        ax.set_ylabel("probability density")

    axes[1].set_xlabel(r"inter-flash interval $T_{FF}$ (s)")
    axes[0].set_xlim(0, X_MAX)
    clean_all(axes)
    fig.tight_layout()
    fig.savefig(FIGS / "B1_ifi_kde.pdf")
    fig.savefig(FIGS / "B1_ifi_kde.png", dpi=200)
    print(f"wrote {FIGS / 'B1_ifi_kde.pdf'}")


if __name__ == "__main__":
    main()
