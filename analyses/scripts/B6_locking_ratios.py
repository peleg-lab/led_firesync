"""B6 -- locking-ratio inspection for 1:2 and 2:1 entrainment.

For every (trial, T_LED) we compute the per-trial locking ratio
    rho = T_FF / T_LED
using two estimators (mean and median) of T_FF over the perturbed segment
(restricted to IFIs in (0.25, 2.5) s and at least 5 valid IFIs per trial).

Reviewer 2 asks whether the LCO-tanh integrate-and-fire model captures 1:2 and
2:1 entrainment.  This first-cut analysis looks at the *experimental* rho
distribution per T_LED and asks: do trials cluster near rho = 1, 1/2, or 2?

Outputs:
  analyses/results/B6_rho_per_trial.csv            (per-trial rho values)
  analyses/results/B6_rho_locking_prevalence.csv   (per-period fraction within
                                                    +/- 5% of rho=1, 1/2, 2)
  analyses/figs/B6_rho_distribution.pdf            (rho stripplot/violin
                                                    per T_LED + reference lines)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data_loader import iter_trials  # noqa: E402
from lib.plot_style import apply_rc, clean  # noqa: E402

apply_rc()

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / "figs"
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

MIN_T = 0.25; MAX_T = 2.5
TOL = 0.05  # +/- 5%: within tolerance of rho=1, 1/2, 2


def main() -> None:
    rows = []
    for trial in iter_trials():
        ff = trial.ff_perturbed
        if ff.size < 6:
            continue
        ifis = np.diff(ff)
        good = ifis[(ifis > MIN_T) & (ifis < MAX_T)]
        if good.size < 5:
            continue
        T_FF_mean = float(np.mean(good))
        T_FF_med = float(np.median(good))
        T_LED = trial.led_period_ms / 1000.0
        rows.append(dict(
            date=trial.date, led_period_ms=trial.led_period_ms, index=trial.index,
            n_ifi=int(good.size),
            T_FF_mean=T_FF_mean, T_FF_median=T_FF_med,
            rho_mean=T_FF_mean / T_LED, rho_median=T_FF_med / T_LED,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "B6_rho_per_trial.csv", index=False)
    print(f"trials with usable rho: {len(df)}")

    # ---- locking prevalence per period ----
    lock_rows = []
    for period in sorted(df["led_period_ms"].unique()):
        sub = df[df["led_period_ms"] == period]["rho_median"].to_numpy()
        n = sub.size
        in_1_1 = int((np.abs(sub - 1.0) < TOL).sum())
        in_2_1 = int((np.abs(sub - 2.0) < TOL).sum())   # T_FF = 2 T_LED -> subharmonic
        in_1_2 = int((np.abs(sub - 0.5) < TOL).sum())   # T_FF = T_LED/2  -> superharmonic
        lock_rows.append(dict(
            led_period_ms=period, n_trials=n,
            frac_1_1=in_1_1 / n, frac_2_1=in_2_1 / n, frac_1_2=in_1_2 / n,
            mean_rho_median=float(sub.mean()),
            median_rho_median=float(np.median(sub)),
        ))
    lock_df = pd.DataFrame(lock_rows)
    lock_df.to_csv(RESULTS / "B6_rho_locking_prevalence.csv", index=False)
    print(lock_df.to_string(index=False))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(10, 5))
    periods = sorted(df["led_period_ms"].unique())
    rng = np.random.default_rng(0)
    for i, period in enumerate(periods):
        vals = df.loc[df["led_period_ms"] == period, "rho_median"].to_numpy()
        x = i + rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(x, vals, s=28, alpha=0.7,
                   color=plt.get_cmap("viridis")(i / max(1, len(periods) - 1)),
                   edgecolor="k", linewidth=0.3)
    for ref, label in [(1.0, "1:1"), (0.5, "1:2 (superharmonic)"),
                       (2.0, "2:1 (subharmonic)")]:
        ax.axhline(ref, ls="--", color="k", alpha=0.5, lw=0.9)
        ax.text(len(periods) - 0.4, ref, f" $\\rho={ref}$ ({label})",
                va="center", fontsize=9)
        ax.fill_between([-0.5, len(periods) - 0.5],
                        ref * (1 - TOL), ref * (1 + TOL),
                        color="k", alpha=0.05)
    ax.set_yscale("log")
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels([f"{p} ms" for p in periods])
    ax.set_xlim(-0.5, len(periods) - 0.5 + 1.4)
    ax.set_xlabel(r"$T_\mathrm{LED}$")
    ax.set_ylabel(r"$\rho = T_{FF}/T_\mathrm{LED}$ (per-trial median)")
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "B6_rho_distribution.pdf")
    fig.savefig(FIGS / "B6_rho_distribution.png", dpi=200)


if __name__ == "__main__":
    main()
