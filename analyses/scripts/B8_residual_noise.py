"""B8 -- effective noise intensity from PRC residuals.

Uses the canonical SELECTION_MODE='midpoint' PRC pipeline (lib.data_loader
.prc_pairs_midpoint).  Same scope and binning as B4.

For each in-scope point Z = 1 - R, we subtract the bin mean and report:
  per-bin residual SD,
  pooled residual SD  (single scalar effective-noise intensity),
  pooled residual variance,
  weighted average per-bin SD.

Outputs:
  analyses/results/B8_residual_noise_per_bin.csv
  analyses/results/B8_residual_noise_summary.csv
  analyses/figs/B8_residual_noise.pdf
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
from lib.data_loader import iter_trials, prc_pairs_midpoint  # noqa: E402
from lib.plot_style import apply_rc, clean_all  # noqa: E402

apply_rc()

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / "figs"
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

N_BINS = 35


def collect_points():
    out = []
    for trial in iter_trials():
        for phi, R, *_ in prc_pairs_midpoint(trial):
            out.append((phi, 1.0 - R))
    return np.array(out)


def main() -> None:
    data = collect_points()
    phi = data[:, 0]; Z = data[:, 1]
    edges = np.linspace(-0.5, 0.5, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(phi, edges) - 1, 0, N_BINS - 1)

    mean_per_bin = np.full(N_BINS, np.nan)
    sd_per_bin = np.full(N_BINS, np.nan)
    n_per_bin = np.zeros(N_BINS, dtype=int)
    residuals = np.full_like(Z, np.nan)
    for b in range(N_BINS):
        mask = bin_idx == b
        if mask.sum() < 2:
            continue
        Zb = Z[mask]
        mean_per_bin[b] = Zb.mean()
        sd_per_bin[b] = Zb.std(ddof=1)
        n_per_bin[b] = int(mask.sum())
        residuals[mask] = Zb - Zb.mean()

    df = pd.DataFrame(dict(
        bin_center=centers, n=n_per_bin,
        mean_Z=mean_per_bin, sd_Z_per_bin=sd_per_bin,
    ))
    df.to_csv(RESULTS / "B8_residual_noise_per_bin.csv", index=False)

    valid = np.isfinite(residuals)
    pooled_var = float(np.mean(residuals[valid] ** 2))
    pooled_sd = float(np.sqrt(pooled_var))
    weighted_sd = float(np.sqrt(
        np.nansum((n_per_bin - 1) * sd_per_bin ** 2) /
        max(1, np.nansum(n_per_bin - 1))))

    summary = pd.DataFrame([
        dict(metric="pooled_residual_sd",  value=pooled_sd),
        dict(metric="pooled_residual_var", value=pooled_var),
        dict(metric="weighted_avg_per_bin_sd", value=weighted_sd),
        dict(metric="n_points", value=int(valid.sum())),
    ])
    summary.to_csv(RESULTS / "B8_residual_noise_summary.csv", index=False)
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    valid_bins = ~np.isnan(sd_per_bin)
    axes[0].bar(centers[valid_bins], sd_per_bin[valid_bins],
                width=0.95 / N_BINS, color="#1f77b4", alpha=0.85,
                edgecolor="white", linewidth=0.5)
    axes[0].axhline(pooled_sd, ls="--", color="crimson", lw=1.2,
                    label=f"pooled residual SD = {pooled_sd:.3f}")
    axes[0].axvline(0.0, ls="--", color="k", alpha=0.4, lw=0.7)
    axes[0].set_xlabel(r"$\phi_\mathrm{eval}$")
    axes[0].set_ylabel(r"residual SD of $Z(\phi)$")
    axes[0].legend()

    axes[1].hist(residuals[valid], bins=61, color="#1f77b4", alpha=0.85,
                 edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, ls="--", color="k", alpha=0.4, lw=0.7)
    axes[1].set_xlabel(r"$Z - \langle Z \rangle_\mathrm{bin}$")
    axes[1].set_ylabel("count")
    axes[1].text(0.04, 0.96,
                 f"$N={int(valid.sum())}$\npooled SD$\\,=\\,{pooled_sd:.3f}$",
                 transform=axes[1].transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="none", alpha=0.85))

    clean_all(axes)
    fig.tight_layout()
    fig.savefig(FIGS / "B8_residual_noise.pdf")
    fig.savefig(FIGS / "B8_residual_noise.png", dpi=200)


if __name__ == "__main__":
    main()
