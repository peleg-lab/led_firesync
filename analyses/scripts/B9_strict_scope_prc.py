"""B9 -- PRC restricted to the strict single-pulse scope, and stratified by
LED-pulse count (Reviewer 2, R2 Major 1).

The manuscript's midpoint-selection PRC keeps every in-scope cycle with >= 1
LED pulse inside T_next and assigns T_next to the pulse nearest the cycle
midpoint.  Here we recompute the binned PRC on:

  * strict subset    : T_prev contains 0 LED pulses AND T_next contains exactly 1
  * n_next = 1       : exactly one pulse in T_next (any number in T_prev)
  * n_next = 2       : two pulses in T_next (the T_next < 2*T_LED filter means
                       >= 3 pulses essentially never occur in scope)
  * n_prev >= 1, n_next = 1 : the complement of the strict subset within n_next = 1

and compare each with the full-scope curve (RMS and max bin-to-bin
deviation, relative to the trial-level bootstrap CI half-widths from B10).

Outputs:
  analyses/results/B9_prc_subsets.csv        binned means per subset
  analyses/results/B9_subset_counts.csv      per-T_LED point counts per subset
  analyses/results/B9_rms_deviation.csv      subset-vs-full deviations
  analyses/results/B9_lobe_stats.csv         lobe means, trial-level CIs, and
                                             T_next / T_prev relative to each
                                             trial's habituation period
  analyses/figs/B9_prc_strict_scope.pdf/.png
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
from lib.data_loader import all_prc_records, iter_trials, bout_segments  # noqa: E402
from lib.plot_style import apply_rc, clean_all  # noqa: E402

apply_rc()
RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / "figs"
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

N_BINS = 35
EDGES = np.linspace(-0.5, 0.5, N_BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
MIN_PER_BIN = 5          # bins with fewer points are left blank
MIN_STRICT_PER_PERIOD = 50
N_BOOT = 2000
RNG = np.random.default_rng(20261007)
LOBE = 0.25              # lobes: 0 < phi < LOBE (advance side), -LOBE < phi < 0 (delay side)


def habituation_period() -> dict:
    """Per-trial median of the bout-censored habituation IFIs (gap <= 2 s,
    >= 3 flashes), the codebase's `prior_period` baseline."""
    out = {}
    for tr in iter_trials():
        hab = tr.ff_habituation
        segs = bout_segments(hab) if hab.size >= 3 else []
        ifis = np.concatenate([np.diff(b) for b in segs]) if segs else np.array([])
        out[tr.path.stem] = float(np.median(ifis)) if ifis.size else np.nan
    return out


def cluster_ci(sub: pd.DataFrame) -> tuple[float, float]:
    """Trial-level bootstrap 95% CI on the mean of R (see B10)."""
    tr = sub.trial.unique(); T = tr.size
    if T < 2:
        return (np.nan, np.nan)
    S = sub.groupby("trial").R.sum().reindex(tr).to_numpy()
    C = sub.groupby("trial").R.size().reindex(tr).to_numpy().astype(float)
    m = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(RNG.integers(0, T, T), minlength=T).astype(float)
        m[b] = (w @ S) / (w @ C)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def binned(df: pd.DataFrame, label: str) -> pd.DataFrame:
    b = np.clip(np.digitize(df["phi"].to_numpy(), EDGES) - 1, 0, N_BINS - 1)
    rows = []
    for k in range(N_BINS):
        r = df["R"].to_numpy()[b == k]
        rows.append(dict(subset=label, bin_center=CENTERS[k], n=int(r.size),
                         mean_R=float(r.mean()) if r.size >= MIN_PER_BIN else np.nan,
                         sem_R=float(r.std(ddof=1) / np.sqrt(r.size)) if r.size >= max(2, MIN_PER_BIN) else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.DataFrame(all_prc_records())
    df["T_hab"] = df.trial.map(habituation_period())
    print(f"in-scope points: {len(df)}  trials: {df.trial.nunique()}")
    print("n_next distribution:", df.n_next.value_counts().sort_index().to_dict())
    print("n_prev distribution:", df.n_prev.clip(upper=4).value_counts().sort_index().to_dict())

    subsets = {
        "full scope (manuscript)": df,
        "strict: 0 in T_prev, 1 in T_next": df[(df.n_prev == 0) & (df.n_next == 1)],
        "exactly 1 in T_next (any T_prev)": df[df.n_next == 1],
        ">=1 in T_prev, 1 in T_next": df[(df.n_prev >= 1) & (df.n_next == 1)],
        "2 in T_next": df[df.n_next == 2],
    }
    # ---- per-T_LED counts ----
    cnt = []
    for p, g in df.groupby("led_period"):
        cnt.append(dict(led_period_ms=p, full=len(g),
                        strict=int(((g.n_prev == 0) & (g.n_next == 1)).sum()),
                        n_next_1=int((g.n_next == 1).sum()),
                        n_next_2=int((g.n_next == 2).sum()),
                        n_prev_ge1_n_next_1=int(((g.n_prev >= 1) & (g.n_next == 1)).sum()),
                        trials=g.trial.nunique(),
                        strict_trials=g[(g.n_prev == 0) & (g.n_next == 1)].trial.nunique()))
    cnt.append(dict(led_period_ms="all", full=len(df),
                    strict=len(subsets["strict: 0 in T_prev, 1 in T_next"]),
                    n_next_1=int((df.n_next == 1).sum()), n_next_2=int((df.n_next == 2).sum()),
                    n_prev_ge1_n_next_1=len(subsets[">=1 in T_prev, 1 in T_next"]),
                    trials=df.trial.nunique(),
                    strict_trials=subsets["strict: 0 in T_prev, 1 in T_next"].trial.nunique()))
    cnt_df = pd.DataFrame(cnt); cnt_df.to_csv(RESULTS / "B9_subset_counts.csv", index=False)
    print(cnt_df.to_string(index=False))

    # ---- binned curves ----
    curves = pd.concat([binned(g, k) for k, g in subsets.items()], ignore_index=True)
    # strict subset per T_LED (where populated)
    strict = subsets["strict: 0 in T_prev, 1 in T_next"]
    per_period = {}
    for p, g in strict.groupby("led_period"):
        if len(g) >= MIN_STRICT_PER_PERIOD:
            per_period[p] = binned(g, f"strict @ {p} ms")
    if per_period:
        curves = pd.concat([curves] + list(per_period.values()), ignore_index=True)
    curves.to_csv(RESULTS / "B9_prc_subsets.csv", index=False)

    # ---- deviations from full scope ----
    full = curves[curves.subset == "full scope (manuscript)"].set_index("bin_center")["mean_R"]
    rms_rows = []
    for k in list(subsets)[1:] + [f"strict @ {p} ms" for p in per_period]:
        c = curves[curves.subset == k].set_index("bin_center")["mean_R"]
        d = (c - full).dropna()
        # sign of lobes preserved?
        pos = c[(c.index > 0.0) & (c.index < 0.25)].mean(); neg = c[(c.index < 0.0) & (c.index > -0.25)].mean()
        rms_rows.append(dict(subset=k, n_points=int(curves[curves.subset == k]["n"].sum()),
                             bins_compared=int(d.size),
                             rms_deviation=float(np.sqrt((d ** 2).mean())),
                             max_abs_deviation=float(d.abs().max()),
                             mean_R_pos_lobe=float(pos), mean_R_neg_lobe=float(neg)))
    rms_df = pd.DataFrame(rms_rows); rms_df.to_csv(RESULTS / "B9_rms_deviation.csv", index=False)
    print(rms_df.to_string(index=False))

    # ---- lobe statistics with trial-level CIs and intrinsic-period baselines ----
    lobe_rows = []
    for k, g in subsets.items():
        for lobe, mask in (("advance side (0 < phi < 0.25)", (g.phi > 0) & (g.phi < LOBE)),
                           ("delay side (-0.25 < phi < 0)", (g.phi < 0) & (g.phi > -LOBE))):
            s = g[mask]; sh = s[s.T_hab.notna()]
            lo, hi = cluster_ci(s)
            lobe_rows.append(dict(subset=k, lobe=lobe, n=len(s), trials=s.trial.nunique(),
                                  mean_R=float(s.R.mean()), ci95_lo=lo, ci95_hi=hi,
                                  median_Tnext_over_Thab=float((sh.T_next / sh.T_hab).median()),
                                  median_Tprev_over_Thab=float((sh.T_prev / sh.T_hab).median()),
                                  median_Tnext_s=float(s.T_next.median()), median_Tprev_s=float(s.T_prev.median())))
    lobe_df = pd.DataFrame(lobe_rows); lobe_df.to_csv(RESULTS / "B9_lobe_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(lobe_df.round(3).to_string(index=False))

    # ---- figure ----
    pretty = {
        "full scope (manuscript)": r"full scope (Figure 3A)",
        "strict: 0 in T_prev, 1 in T_next": r"strict: 0 pulses in $T_{\mathrm{prev}}$, 1 in $T_{\mathrm{next}}$",
        "exactly 1 in T_next (any T_prev)": r"1 pulse in $T_{\mathrm{next}}$, any in $T_{\mathrm{prev}}$",
        ">=1 in T_prev, 1 in T_next": r"$\geq 1$ pulse in $T_{\mathrm{prev}}$, 1 in $T_{\mathrm{next}}$",
        "2 in T_next": r"2 pulses in $T_{\mathrm{next}}$",
    }
    styles = {
        "full scope (manuscript)": dict(color="#1f77b4", lw=2.4, ms=4.0, zorder=6),
        "exactly 1 in T_next (any T_prev)": dict(color="#2ca02c", lw=1.4, ms=3.0, zorder=5),
        "strict: 0 in T_prev, 1 in T_next": dict(color="#d62728", lw=2.0, ms=3.5, zorder=7),
        ">=1 in T_prev, 1 in T_next": dict(color="#9467bd", lw=1.4, ms=3.0, zorder=4),
        "2 in T_next": dict(color="#9a9a9a", lw=1.1, ms=2.5, ls="--", zorder=3),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    ax = axes[0]
    for k, st in styles.items():
        c = curves[curves.subset == k]
        n_tot = int(c["n"].sum())
        ax.errorbar(c.bin_center, c.mean_R, yerr=c.sem_R, fmt="-o", capsize=0, elinewidth=0.8,
                    alpha=0.95, label=f"{pretty[k]}  ($n={n_tot:,}$)", **st)
    ax.axhline(1.0, ls="--", color="k", alpha=0.4, lw=0.8); ax.axvline(0.0, ls=":", color="k", alpha=0.4, lw=0.8)
    ax.set_xlabel(r"$\phi_{\mathrm{eval}}$"); ax.set_ylabel(r"$R(\phi_{\mathrm{eval}}) = T_{\mathrm{next}}/T_{\mathrm{prev}}$")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.3, 1.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5, frameon=False)
    ax.text(-0.12, 1.02, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    ax = axes[1]
    c = curves[curves.subset == "full scope (manuscript)"]
    ax.plot(c.bin_center, c.mean_R, "-", color="k", lw=2.4, label="full scope, all $T_{LED}$ pooled", zorder=6)
    cmap = plt.get_cmap("viridis_r"); periods = sorted(per_period)
    for i, p in enumerate(periods):
        c = per_period[p]
        ax.errorbar(c.bin_center, c.mean_R, yerr=c.sem_R, fmt="-o", ms=3.2, capsize=0, lw=1.3, elinewidth=0.8,
                    color=cmap(i / max(1, len(periods) - 1)), alpha=0.95,
                    label=f"strict subset, $T_{{LED}}={p}$ ms  ($n={int(c['n'].sum())}$)")
    ax.axhline(1.0, ls="--", color="k", alpha=0.4, lw=0.8); ax.axvline(0.0, ls=":", color="k", alpha=0.4, lw=0.8)
    ax.set_xlabel(r"$\phi_{\mathrm{eval}}$"); ax.set_xlim(-0.5, 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5, frameon=False)
    ax.text(-0.06, 1.02, "B", transform=ax.transAxes, fontsize=14, fontweight="bold")
    clean_all(axes); fig.tight_layout()
    fig.savefig(FIGS / "B9_prc_strict_scope.pdf"); fig.savefig(FIGS / "B9_prc_strict_scope.png", dpi=200)


if __name__ == "__main__":
    main()
