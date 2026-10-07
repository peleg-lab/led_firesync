"""B2 -- distribution of LED pulses per firefly IFI.

For every successive pair of firefly onsets (t_i, t_{i+1}) inside the perturbed
segment of each trial we count the LED pulses falling in (t_i, t_{i+1}).  We
also count, for *pairs* of successive intervals (T_prev, T_next), the fraction
that satisfy Reviewer 3's reading of the manuscript's PRC scope:

    T_prev contains 0 LED pulses  AND  T_next contains exactly 1 LED pulse.

Outputs:
  analyses/results/B2_pulses_per_ifi.csv             (counts per LED period)
  analyses/results/B2_scope_fraction.csv             (single-pulse scope fraction)
  analyses/figs/B2_pulses_per_ifi.pdf                (stacked-bar by LED period)
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
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

BINS = ["0", "1", "2", "3", "4+"]


def bucket(n: int) -> str:
    if n >= 4:
        return "4+"
    return str(n)


def main() -> None:
    per_period_counts: dict[int, Counter] = defaultdict(Counter)
    pair_total = Counter()
    pair_scope = Counter()   # (T_prev=0, T_next=1) matches

    for trial in iter_trials():
        ff = trial.ff_perturbed
        led = trial.led_onsets
        if ff.size < 2 or led.size == 0:
            continue
        # interval-by-interval pulse counts
        pulse_counts = np.array([
            int(((led > ff[i]) & (led < ff[i + 1])).sum())
            for i in range(ff.size - 1)
        ])
        for n in pulse_counts:
            per_period_counts[trial.led_period_ms][bucket(n)] += 1
        # consecutive-pair check
        for j in range(pulse_counts.size - 1):
            pair_total[trial.led_period_ms] += 1
            if pulse_counts[j] == 0 and pulse_counts[j + 1] == 1:
                pair_scope[trial.led_period_ms] += 1

    # ---- per-period interval counts ----
    rows = []
    for period in sorted(per_period_counts):
        c = per_period_counts[period]
        total = sum(c.values())
        for b in BINS:
            rows.append({
                "led_period_ms": period, "pulses_per_ifi": b,
                "count": c.get(b, 0), "fraction": (c.get(b, 0) / total) if total else 0.0,
                "total_intervals": total,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "B2_pulses_per_ifi.csv", index=False)

    # ---- scope-pair fraction ----
    scope_rows = []
    for period in sorted(pair_total):
        scope_rows.append({
            "led_period_ms": period,
            "scope_pairs": pair_scope[period],
            "total_pairs": pair_total[period],
            "scope_fraction": pair_scope[period] / pair_total[period] if pair_total[period] else 0.0,
        })
    overall_scope = sum(pair_scope.values()) / sum(pair_total.values())
    scope_rows.append({
        "led_period_ms": "all", "scope_pairs": sum(pair_scope.values()),
        "total_pairs": sum(pair_total.values()), "scope_fraction": overall_scope,
    })
    scope_df = pd.DataFrame(scope_rows)
    scope_df.to_csv(RESULTS / "B2_scope_fraction.csv", index=False)
    print("== pulses-per-IFI (overall fraction across all periods) ==")
    print(df.groupby("pulses_per_ifi")["count"].sum().pipe(lambda s: s / s.sum()).round(3))
    print("\n== scope (T_prev=0, T_next=1) pair fraction by period ==")
    print(scope_df.to_string(index=False))

    # ---- figure: stacked-bar by LED period ----
    periods = sorted(per_period_counts)
    bottom = np.zeros(len(periods))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(BINS):
        vals = np.array([per_period_counts[p].get(b, 0) /
                         sum(per_period_counts[p].values()) for p in periods])
        ax.bar(range(len(periods)), vals, bottom=bottom,
               label=f"{b} LED pulse(s)", color=cmap(i / (len(BINS) - 1)))
        bottom += vals

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels([f"{p} ms" for p in periods])
    ax.set_xlabel("nominal $T_\\mathrm{LED}$ (ms)")
    ax.set_ylabel("fraction of firefly inter-flash intervals")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_ylim(0, 1.0)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "B2_pulses_per_ifi.pdf")
    fig.savefig(FIGS / "B2_pulses_per_ifi.png", dpi=200)


if __name__ == "__main__":
    main()
