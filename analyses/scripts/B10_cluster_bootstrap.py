"""B10 -- trial-level (cluster) bootstrap for the PRC confidence bands
(Reviewer 2, R2 Major 2).  Supersedes the pooled-point bootstrap in B4.

Each trial is one wild-caught individual (one firefly per trial, released
after a single trial), so the trial is simultaneously the individual-level
unit; no higher level of nesting exists.  We resample the N_trial trials that
contribute in-scope PRC points with replacement (N_BOOT times), pool the
cycles of the resampled trials with multiplicity, and recompute every per-bin
mean; the 2.5/97.5 percentiles give the cluster-bootstrap CI.  The pooled
point-level bootstrap of B4 is recomputed on the same draws for comparison.

Outputs:
  analyses/results/B10_prc_cluster_bands.csv
  analyses/results/B10_ci_summary.csv
  analyses/figs/B10_prc_variance_cluster.pdf/.png   (replacement for Fig. S6)
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
from lib.data_loader import all_prc_records  # noqa: E402
from lib.plot_style import apply_rc, clean_all  # noqa: E402

apply_rc()
RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / "figs"
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

N_BINS = 35
N_BOOT = 2000
RNG = np.random.default_rng(20261007)
EDGES = np.linspace(-0.5, 0.5, N_BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])


def main() -> None:
    df = pd.DataFrame(all_prc_records())
    df["bin"] = np.clip(np.digitize(df["phi"].to_numpy(), EDGES) - 1, 0, N_BINS - 1)
    trials = sorted(df.trial.unique()); T = len(trials); tix = {t: i for i, t in enumerate(trials)}
    df["t"] = df.trial.map(tix)
    print(f"points {len(df)}  trials contributing {T}")
    share = df.groupby("trial").size().sort_values(ascending=False)
    print(f"largest single-trial share: {share.iloc[0]} points = {100*share.iloc[0]/len(df):.1f}%; "
          f"median {share.median():.0f} points/trial")

    # per-trial, per-bin sums and counts
    S = np.zeros((T, N_BINS)); C = np.zeros((T, N_BINS))
    np.add.at(S, (df.t.to_numpy(), df.bin.to_numpy()), df.R.to_numpy())
    np.add.at(C, (df.t.to_numpy(), df.bin.to_numpy()), 1.0)
    mean_R = S.sum(0) / C.sum(0)
    n_bin = C.sum(0).astype(int); trials_per_bin = (C > 0).sum(0)

    # --- cluster bootstrap: resample trials with replacement ---
    boot = np.empty((N_BOOT, N_BINS))
    for b in range(N_BOOT):
        w = np.bincount(RNG.integers(0, T, size=T), minlength=T).astype(float)
        boot[b] = (w @ S) / (w @ C)
    lo_c, hi_c = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)

    # --- pooled point-level bootstrap (B4 procedure) on the same bins ---
    lo_p = np.empty(N_BINS); hi_p = np.empty(N_BINS); sd = np.empty(N_BINS)
    for k in range(N_BINS):
        r = df.R.to_numpy()[df.bin.to_numpy() == k]
        sd[k] = r.std(ddof=1)
        idx = RNG.integers(0, r.size, size=(N_BOOT, r.size))
        m = r[idx].mean(axis=1)
        lo_p[k], hi_p[k] = np.percentile(m, 2.5), np.percentile(m, 97.5)

    out = pd.DataFrame(dict(bin_center=CENTERS, n=n_bin, n_trials=trials_per_bin, mean_R=mean_R, sd_R=sd,
                            lo95_R_pooled=lo_p, hi95_R_pooled=hi_p, lo95_R_cluster=lo_c, hi95_R_cluster=hi_c))
    out["mean_Z"] = 1 - out.mean_R; out["sd_Z"] = out.sd_R
    out["lo95_Z_cluster"] = 1 - out.hi95_R_cluster; out["hi95_Z_cluster"] = 1 - out.lo95_R_cluster
    out["lo95_Z_pooled"] = 1 - out.hi95_R_pooled; out["hi95_Z_pooled"] = 1 - out.lo95_R_pooled
    out["halfwidth_pooled"] = 0.5 * (out.hi95_R_pooled - out.lo95_R_pooled)
    out["halfwidth_cluster"] = 0.5 * (out.hi95_R_cluster - out.lo95_R_cluster)
    out["ratio_cluster_to_pooled"] = out.halfwidth_cluster / out.halfwidth_pooled
    out.to_csv(RESULTS / "B10_prc_cluster_bands.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["bin_center", "n", "n_trials", "mean_R", "sd_R", "halfwidth_pooled", "halfwidth_cluster", "ratio_cluster_to_pooled"]].round(3).to_string(index=False))

    # lobe resolution: does the cluster CI exclude R = 1 in the lobes?
    pos = (CENTERS > 0) & (CENTERS < 0.3); neg = (CENTERS < 0) & (CENTERS > -0.3)
    summ = dict(
        n_points=len(df), n_trials=T, n_boot=N_BOOT,
        sd_min=sd.min(), sd_max=sd.max(),
        hw_pooled_min=out.halfwidth_pooled.min(), hw_pooled_max=out.halfwidth_pooled.max(),
        hw_cluster_min=out.halfwidth_cluster.min(), hw_cluster_max=out.halfwidth_cluster.max(),
        ratio_min=out.ratio_cluster_to_pooled.min(), ratio_median=out.ratio_cluster_to_pooled.median(),
        ratio_max=out.ratio_cluster_to_pooled.max(),
        pos_lobe_bins_excluding_1=int((hi_c[pos] < 1).sum()), pos_lobe_bins=int(pos.sum()),
        neg_lobe_bins_excluding_1=int((lo_c[neg] > 1).sum()), neg_lobe_bins=int(neg.sum()),
        n_bins_with_cluster_ci_excluding_1=int(((hi_c < 1) | (lo_c > 1)).sum()),
        n_bins_with_pooled_ci_excluding_1=int(((hi_p < 1) | (lo_p > 1)).sum()),
        frac_phi_pre_ge1=float((df.phi_pre >= 1).mean()),   # cycles where the LED arrived after the expected flash
    )
    pd.DataFrame([summ]).to_csv(RESULTS / "B10_ci_summary.csv", index=False)
    for k, v in summ.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    # ---- figure (same layout as B4 / Fig. S6) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, letter, y, loc, hic, lop, hip, ylab, base in (
        (axes[0], "A", out.mean_R, out.lo95_R_cluster, out.hi95_R_cluster, out.lo95_R_pooled, out.hi95_R_pooled,
         r"$R(\phi_{\mathrm{eval}}) = T_{\mathrm{next}}/T_{\mathrm{prev}}$", 1.0),
        (axes[1], "B", out.mean_Z, out.lo95_Z_cluster, out.hi95_Z_cluster, out.lo95_Z_pooled, out.hi95_Z_pooled,
         r"$Z(\phi_{\mathrm{eval}}) = 1 - R(\phi_{\mathrm{eval}})$", 0.0),
    ):
        x = out.bin_center.to_numpy()
        ax.fill_between(x, y - out.sd_R, y + out.sd_R, color="#1f77b4", alpha=0.13, lw=0, label=r"$\pm$ 1 SD of cycles in bin")
        ax.fill_between(x, loc, hic, color="#1f77b4", alpha=0.45, lw=0, label="95% CI, trial-level bootstrap (110 trials)")
        ax.plot(x, lop, ":", color="#d62728", lw=1.3, label="95% CI, pooled-cycle bootstrap (for comparison)")
        ax.plot(x, hip, ":", color="#d62728", lw=1.3)
        ax.plot(x, y, "-o", color="#1f77b4", lw=1.6, ms=3.5, label="bin mean")
        ax.axhline(base, ls="--", color="k", alpha=0.4, lw=0.8); ax.axvline(0.0, ls=":", color="k", alpha=0.4, lw=0.8)
        ax.set_xlabel(r"$\phi_{\mathrm{eval}}$"); ax.set_ylabel(ylab); ax.set_xlim(-0.5, 0.5)
        ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    clean_all(axes); fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.savefig(FIGS / "B10_prc_variance_cluster.pdf"); fig.savefig(FIGS / "B10_prc_variance_cluster.png", dpi=200)


if __name__ == "__main__":
    main()
