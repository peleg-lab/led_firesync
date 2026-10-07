"""B3 -- PRC sensitivity to alternative free-running predictors.

Reviewer 3 asks why we set p_prev = T_{i-1} after the first IFI in each bout
(equivalently T_predicted = previous IFI).  We recompute the PRC under three
predictor choices, holding the rest of the codebase's bout pipeline fixed
(same bouts, same phi_eval per Eq. (4), same closest-LED rule):

    pred = T_{i-1}                 ("Tprev", paper / codebase choice)
    pred = mean(habituation IFIs)  ("habituation mean", per-trial constant)
    pred = mean(previous n IFIs)   ("trailing 3-cycle mean", lagged predictor on bout series)

For all three the first IFI in each bout uses prior_period (habituation
median) -- this is the codebase's initialization -- so the only thing that
varies is the rule for choosing the denominator on subsequent IFIs.

Outputs:
  analyses/results/B3_prc_binned_means.csv
  analyses/results/B3_rms_deviation.csv
  analyses/figs/B3_predictor_sensitivity.pdf
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
from lib.data_loader import (  # noqa: E402
    iter_trials, interflash_intervals, prc_pairs_midpoint,
)
from lib.plot_style import apply_rc, clean  # noqa: E402

apply_rc()

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / "figs"
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

N_BINS = 35
N_AR = 3
MIN_T = 0.25   # bout-censoring window (matches STAR Methods PRC pipeline)
MAX_T = 2.5


def collect_records():
    """For each midpoint-mode PRC point keep three candidate predictors of
    T_next so that R3's three alternatives can be compared on identical scope:

        T_prev           = paper / codebase choice (the cycle immediately before
                           the perturbed one).
        habituation mean = arithmetic mean of the WITHIN-BOUT habituation IFIs
                           (i.e. the same bout-censored intervals the rest of
                           the pipeline uses; IFIs outside (MIN_T, MAX_T) are
                           dropped so inter-bout silences don't inflate the
                           denominator).  This is the fair comparison with the
                           reviewer's option (a): "mean of all intervals
                           collected during the 72 s of unperturbed activity"
                           read consistently with the manuscript's bout-censoring
                           convention.
        AR(N_AR)         = arithmetic mean of the N_AR IFIs in the full firefly
                           stream immediately preceding t_prev, ONLY when all
                           N_AR of those IFIs are themselves in (MIN_T, MAX_T).
                           Records where any of the N_AR preceding IFIs falls
                           outside that window (e.g. the perturbed cycle sits
                           at the start of a bout and one of the preceding IFIs
                           is an inter-bout silence) are skipped for the AR
                           column so we don't mix bout and inter-bout intervals
                           into the denominator.
    """
    rows = []
    for trial in iter_trials():
        hab_ifis = interflash_intervals(trial.ff_habituation)
        # (a) raw habituation mean: literal reading of the reviewer's suggestion,
        #     "mean of all the intervals collected during the ~72 s of unperturbed
        #     activity."  Includes inter-bout silences.
        if hab_ifis.size < 3:
            continue
        hab_mean_raw = float(np.mean(hab_ifis))
        # (b) bout-censored habituation mean: same idea but restricted to within-
        #     bout IFIs, matching the rest of the pipeline's bout-censoring
        #     convention.  This is the fair apples-to-apples comparison against
        #     T_prev.
        hab_ifis_within = hab_ifis[(hab_ifis > MIN_T) & (hab_ifis < MAX_T)]
        if hab_ifis_within.size < 3:
            continue
        hab_mean_filt = float(np.mean(hab_ifis_within))
        ff_all = trial.ff_onsets
        ifi_all = np.diff(ff_all)  # IFI[k] = ff_all[k+1] - ff_all[k]
        for phi, R, T_prev, T_next, idx_prev in prc_pairs_midpoint(trial):
            # IFIs immediately preceding t_prev = ff_all[idx_prev]:
            # IFI[idx_prev - 1] = T_prev, IFI[idx_prev - 2], ..., IFI[idx_prev - N_AR].
            lo = idx_prev - N_AR
            hi = idx_prev          # exclusive upper bound on IFI index
            if lo < 0 or hi <= lo:
                continue           # need N_AR true preceding IFIs
            preceding = ifi_all[lo:hi]
            if not np.all((preceding > MIN_T) & (preceding < MAX_T)):
                continue           # skip AR when a preceding IFI is inter-bout
            T_ar = float(np.mean(preceding))
            rows.append((phi, T_next, T_prev, hab_mean_raw, hab_mean_filt, T_ar))
    return rows


def bin_curves(records):
    df = pd.DataFrame(records,
                      columns=["phi", "Tcurrent", "Tprev",
                               "Thab_raw", "Thab_filt", "Tar"])
    df["R_prev"]      = df["Tcurrent"] / df["Tprev"]
    df["R_hab_raw"]   = df["Tcurrent"] / df["Thab_raw"]
    df["R_hab_filt"]  = df["Tcurrent"] / df["Thab_filt"]
    df["R_ar"]        = df["Tcurrent"] / df["Tar"]

    edges = np.linspace(-0.5, 0.5, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    df["bin"] = pd.cut(df["phi"], edges, labels=False, include_lowest=True)

    out = []
    for predictor, col in [
        ("Tprev",                    "R_prev"),
        ("trailing 3-cycle mean",                    "R_ar"),
        ("habituation mean (filt)",  "R_hab_filt"),
        ("habituation mean (raw)",   "R_hab_raw"),
    ]:
        g = df.groupby("bin")[col]
        out.append(pd.DataFrame({
            "bin_center": centers,
            "predictor": predictor,
            "mean_R": g.mean().reindex(range(N_BINS)).to_numpy(),
            "sem_R":  g.sem().reindex(range(N_BINS)).to_numpy(),
            "n":      g.size().reindex(range(N_BINS)).fillna(0).astype(int).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True), df


def main() -> None:
    records = collect_records()
    print(f"in-scope records: {len(records)}")
    binned, df = bin_curves(records)
    binned.to_csv(RESULTS / "B3_prc_binned_means.csv", index=False)

    pivot = binned.pivot(index="bin_center", columns="predictor", values="mean_R")
    rms_rows = []
    cols = pivot.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            d = pivot[cols[i]] - pivot[cols[j]]
            rms_rows.append({"pair": f"{cols[i]} vs {cols[j]}",
                             "rms_deviation": float(np.sqrt(np.nanmean(d ** 2))),
                             "max_abs_deviation": float(np.nanmax(np.abs(d)))})
    rms_df = pd.DataFrame(rms_rows)
    rms_df.to_csv(RESULTS / "B3_rms_deviation.csv", index=False)
    print(rms_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {
        "Tprev":                    "#1f77b4",   # blue
        "trailing 3-cycle mean":                    "#2ca02c",   # green
        "habituation mean (filt)":  "#ff7f0e",   # orange
        "habituation mean (raw)":   "#7f7f7f",   # gray, dashed
    }
    labels = {
        "Tprev":                    r"$T_\mathrm{prev}$ (paper)",
        "trailing 3-cycle mean":                    "trailing 3-cycle mean",
        "habituation mean (filt)":  "habituation mean, bout-censored",
        "habituation mean (raw)":   "habituation mean, raw (includes inter-bout gaps)",
    }
    linestyles = {
        "Tprev":                    "-",
        "trailing 3-cycle mean":                    "-",
        "habituation mean (filt)":  "-",
        "habituation mean (raw)":   "--",
    }
    for predictor in ["Tprev", "trailing 3-cycle mean",
                      "habituation mean (filt)", "habituation mean (raw)"]:
        sub = binned[binned["predictor"] == predictor]
        ax.errorbar(sub["bin_center"], sub["mean_R"], yerr=sub["sem_R"],
                    fmt="o", color=colors[predictor], label=labels[predictor],
                    capsize=2, lw=1.4, ms=3.5, alpha=0.9,
                    linestyle=linestyles[predictor])
    ax.axhline(1.0, color="k", ls="--", alpha=0.4, lw=0.7)
    ax.axvline(0.0, color="k", ls="--", alpha=0.4, lw=0.7)
    ax.set_xlabel(r"$\phi_\mathrm{eval}$")
    ax.set_ylabel(r"$R(\phi_\mathrm{eval}) = T_\mathrm{next}/T_\mathrm{predicted}$")
    ax.legend(loc="best", fontsize=9)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "B3_predictor_sensitivity.pdf")
    fig.savefig(FIGS / "B3_predictor_sensitivity.png", dpi=200)


if __name__ == "__main__":
    main()
