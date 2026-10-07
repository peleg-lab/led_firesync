"""B4 -- PRC variance bands (SD + bootstrap 95% CI).

Uses the canonical SELECTION_MODE='midpoint' PRC pipeline from the codebase
(`analyze_firefly_led_synchronization` in helpers/plotting_helpers.py),
factored into `lib.data_loader.prc_pairs_midpoint`:

  * consecutive triples (t_prevprev, t_prev, t_next) within bouts (gap<=2s,
    bout flash-count diff >= 5);
  * filters T_prev,T_next > 0.30 s, T_next < 2 * T_LED;
  * LED inside (t_prev, t_next) chosen as the one closest to the midpoint;
  * phi = wrap_half((L - t_prev) / T_prev);
  * R = T_next / T_prev, Z = 1 - R.

Binning uses N_BINS = 35 (odd, matches codebase) so a bin is centered at phi=0.
Per bin we report mean, SD, 95% bootstrap CI for R and Z.

Outputs:
  analyses/results/B4_prc_bands.csv
  analyses/figs/B4_prc_variance_bands.pdf
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
N_BOOT = 2000
RNG = np.random.default_rng(20260519)


def collect_points():
    out = []
    for trial in iter_trials():
        for phi, R, *_ in prc_pairs_midpoint(trial):
            out.append((phi, R))
    return np.array(out)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    if values.size < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    data = collect_points()
    print(f"PRC scope points: {data.shape[0]}")
    phi = data[:, 0]; R = data[:, 1]; Z = 1.0 - R
    edges = np.linspace(-0.5, 0.5, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(phi, edges) - 1, 0, N_BINS - 1)

    rows = []
    for b in range(N_BINS):
        mask = bin_idx == b
        Rb = R[mask]; Zb = Z[mask]
        if Rb.size < 2:
            rows.append(dict(bin_center=centers[b], n=int(Rb.size),
                             mean_R=np.nan, sd_R=np.nan, lo95_R=np.nan, hi95_R=np.nan,
                             mean_Z=np.nan, sd_Z=np.nan, lo95_Z=np.nan, hi95_Z=np.nan))
            continue
        loR, hiR = bootstrap_ci(Rb); loZ, hiZ = bootstrap_ci(Zb)
        rows.append(dict(
            bin_center=centers[b], n=int(Rb.size),
            mean_R=float(Rb.mean()), sd_R=float(Rb.std(ddof=1)),
            lo95_R=loR, hi95_R=hiR,
            mean_Z=float(Zb.mean()), sd_Z=float(Zb.std(ddof=1)),
            lo95_Z=loZ, hi95_Z=hiZ,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "B4_prc_bands.csv", index=False)
    print(df[["bin_center", "n", "mean_R", "sd_R", "lo95_R", "hi95_R"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for ax, ycol, sdcol, locol, hicol, ylab, baseline in (
        (axes[0], "mean_R", "sd_R", "lo95_R", "hi95_R",
         r"$R(\phi_{\mathrm{eval}}) = T_{\mathrm{next}}/T_{\mathrm{prev}}$", 1.0),
        (axes[1], "mean_Z", "sd_Z", "lo95_Z", "hi95_Z",
         r"$Z(\phi_{\mathrm{eval}}) = 1 - R(\phi_{\mathrm{eval}})$", 0.0),
    ):
        valid = df[ycol].notna()
        x = df.loc[valid, "bin_center"].to_numpy()
        y = df.loc[valid, ycol].to_numpy()
        sd = df.loc[valid, sdcol].to_numpy()
        lo = df.loc[valid, locol].to_numpy()
        hi = df.loc[valid, hicol].to_numpy()
        ax.fill_between(x, y - sd, y + sd, color="#1f77b4", alpha=0.15,
                        label=r"$\pm$ 1 SD")
        ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.40,
                        label="95\\% bootstrap CI")
        ax.plot(x, y, "-o", color="#1f77b4", lw=1.5, ms=3.5, label="bin mean")
        ax.axhline(baseline, ls="--", color="k", alpha=0.4, lw=0.7)
        ax.axvline(0.0, ls="--", color="k", alpha=0.4, lw=0.7)
        ax.set_xlabel(r"$\phi_{\mathrm{eval}}$")
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=9)
    clean_all(axes)
    fig.tight_layout()
    fig.savefig(FIGS / "B4_prc_variance_bands.pdf")
    fig.savefig(FIGS / "B4_prc_variance_bands.png", dpi=200)


if __name__ == "__main__":
    main()
