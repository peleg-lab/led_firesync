"""B5 -- habituation duration vs perturbed-T_FF correlation.

For each trial we compute
    h        = habituation duration (time from t=0 to first LED pulse, s)
    T_mean   = mean T_FF in the perturbed segment, restricted to (0.25, 2.5) s
    T_med    = median T_FF in the perturbed segment, same restriction
    A_R      = PRC amplitude proxy = median |R - 1| across all in-scope intervals
               (R = T_next/T_prev for perturbed IFIs containing >= 1 LED pulse)
    T_hab_mean_bs, T_hab_med_bs
             = mean / median T_FF in the *habituation* segment, computed after
               bootstrap-downsampling every trial's habituation IFI sample to
               a common size (N = min hab-IFI count across included trials).
               Motivated by the concern that trials with longer habituation
               have more precise per-trial estimates and that this precision
               could confound the reported correlation; downsampling equalizes
               statistical precision.

We then report Pearson and Spearman correlations, each with a 95% CI estimated
by Fisher's z-transform.

Outputs:
  analyses/results/B5_habituation_correlation.csv   (per-trial table)
  analyses/results/B5_correlation_summary.csv       (Pearson/Spearman + 95% CI)
  analyses/figs/B5_habituation_scatter.pdf          (5-panel scatter w/ fits)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data_loader import iter_trials  # noqa: E402
from lib.plot_style import apply_rc, clean_all  # noqa: E402

apply_rc()

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / "figs"
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

MIN_T = 0.25; MAX_T = 2.5
BOOTSTRAP_B = 500                      # bootstrap resamples per trial
BOOTSTRAP_SEED = 20260701
BOOTSTRAP_MIN_HAB_IFIS = 3             # skip trials with fewer than this many hab IFIs
                                       # (so the common downsample size is at least 3)


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n < 4 or not (-1 < r < 1):
        return (np.nan, np.nan)
    z = np.arctanh(r); se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for trial in iter_trials():
        h = trial.habituation_duration
        if not (np.isfinite(h) and h > 0):
            continue
        ff = trial.ff_perturbed; led = trial.led_onsets
        if ff.size < 4 or led.size == 0:
            continue
        # ---- habituation IFIs (raw, all bout-filter-eligible) ----
        hab_ff = trial.ff_habituation
        hab_ifis_all = np.diff(hab_ff) if hab_ff.size >= 2 else np.array([])
        hab_ifis = hab_ifis_all[(hab_ifis_all > MIN_T) & (hab_ifis_all < MAX_T)]
        if hab_ifis.size < BOOTSTRAP_MIN_HAB_IFIS:
            continue
        # ---- perturbed IFIs ----
        ifis = np.diff(ff)
        good = (ifis > MIN_T) & (ifis < MAX_T)
        if good.sum() < 3:
            continue
        T_mean = float(np.mean(ifis[good]))
        T_med = float(np.median(ifis[good]))
        # PRC amplitude proxy: median |R-1| over perturbed IFIs containing >=1 LED pulse
        R_vals = []
        for j in range(1, ifis.size):
            if not (MIN_T < ifis[j - 1] < MAX_T and MIN_T < ifis[j] < MAX_T):
                continue
            n_pulses = int(((led > ff[j]) & (led < ff[j + 1])).sum())
            if n_pulses < 1:
                continue
            R_vals.append(ifis[j] / ifis[j - 1])
        if len(R_vals) < 3:
            continue
        A_R = float(np.median(np.abs(np.array(R_vals) - 1.0)))
        rows.append(dict(
            date=trial.date, led_period_ms=trial.led_period_ms,
            index=trial.index, habituation_s=h, T_mean=T_mean,
            T_median=T_med, A_R=A_R, n_R=len(R_vals),
            hab_ifis=hab_ifis,  # kept for the bootstrap step below
            n_hab_ifis=int(hab_ifis.size),
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no usable trials")

    # ---- Bootstrap-downsample habituation T_FF to a common sample size ----
    N_common = int(df["n_hab_ifis"].min())
    print(f"bootstrap common sample size N_common = {N_common} (min hab-IFI count)")
    boot_means, boot_meds = [], []
    for _, row in df.iterrows():
        ifis_i = np.asarray(row["hab_ifis"])
        # Bootstrap: draw N_common samples WITH REPLACEMENT, repeat B times.
        # (With replacement so that trials with exactly N_common IFIs still
        # participate in the bootstrap distribution.)
        idx = rng.integers(0, ifis_i.size, size=(BOOTSTRAP_B, N_common))
        samples = ifis_i[idx]
        boot_means.append(float(samples.mean(axis=1).mean()))
        boot_meds.append(float(np.median(samples, axis=1).mean()))
    df["T_hab_mean_bs"] = boot_means
    df["T_hab_med_bs"] = boot_meds
    df = df.drop(columns=["hab_ifis"])
    df.to_csv(RESULTS / "B5_habituation_correlation.csv", index=False)
    print(f"trials with usable habituation + perturbed data: {len(df)}")

    # ---- Pearson / Spearman across all trials ----
    summary = []
    for ylabel, ycol in [
        ("T_mean (perturbed)", "T_mean"),
        ("T_median (perturbed)", "T_median"),
        ("|R-1| amplitude", "A_R"),
        ("T_hab_mean (bootstrap)", "T_hab_mean_bs"),
        ("T_hab_median (bootstrap)", "T_hab_med_bs"),
    ]:
        x = df["habituation_s"].to_numpy(); y = df[ycol].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        n = x.size
        pr = stats.pearsonr(x, y); sr = stats.spearmanr(x, y)
        pr_lo, pr_hi = fisher_ci(pr.statistic, n)
        sr_lo, sr_hi = fisher_ci(sr.statistic, n)
        summary.append(dict(metric=ylabel, n=int(n),
                            pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
                            pearson_lo95=pr_lo, pearson_hi95=pr_hi,
                            spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue),
                            spearman_lo95=sr_lo, spearman_hi95=sr_hi))
    out = pd.DataFrame(summary)
    out.to_csv(RESULTS / "B5_correlation_summary.csv", index=False)
    print(out.to_string(index=False))

    # ---- figure: 5-panel scatter w/ fits (3 perturbed + 2 bootstrap-hab) ----
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    panels = [
        (axes[0], r"mean perturbed $T_{FF}$ (s)", "T_mean"),
        (axes[1], r"median perturbed $T_{FF}$ (s)", "T_median"),
        (axes[2], r"median $|R-1|$", "A_R"),
        (axes[3], r"bootstrap-mean habituation $T_{FF}$ (s)", "T_hab_mean_bs"),
        (axes[4], r"bootstrap-median habituation $T_{FF}$ (s)", "T_hab_med_bs"),
    ]
    for ax, ylabel, ycol in panels:
        x = df["habituation_s"].to_numpy(); y = df[ycol].to_numpy()
        sc = ax.scatter(x, y, c=df["led_period_ms"], cmap="viridis", s=22,
                        alpha=0.85, edgecolor="white", linewidth=0.4)
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="crimson", lw=1.2, alpha=0.9)
        pr = stats.pearsonr(x, y)
        r2 = pr.statistic ** 2
        ax.text(0.04, 0.96,
                f"$r^2 = {r2:.3f}$\n$p = {pr.pvalue:.2g}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.85))
        ax.set_xlabel("habituation duration (s)")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.03, y=0.08)
    clean_all(axes)

    # Colourbar to the right of the third panel
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.18, top=0.95, wspace=0.32)
    cax = fig.add_axes([0.93, 0.18, 0.012, 0.77])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"$T_\mathrm{LED}$ (ms)")
    cbar.outline.set_visible(False)

    fig.savefig(FIGS / "B5_habituation_scatter.pdf")
    fig.savefig(FIGS / "B5_habituation_scatter.png", dpi=200)


if __name__ == "__main__":
    main()
