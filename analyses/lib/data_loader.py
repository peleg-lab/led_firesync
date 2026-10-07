"""Shared data loader for B1--B8 analyses.

Each CSV in led_firesync/data_paths/ holds a 60-fps state timeseries:
    column 0: "(timestamp, ff_state)"   ff_state in {0, 1}
    column 1: "(timestamp, led_state)"  led_state in {1, 2} (1=off, 2=on)

We deduce *onsets* by detecting rising edges (0->1 for FF, 1->2 for LED) and
return per-trial dicts with everything downstream analyses need.
"""
from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]   # led_firesync/ (repo root)
DATA_DIR = REPO_ROOT / "data_paths"

FILE_RE = re.compile(r"^(\d{8})_(\d+)_(\d+)\.csv$")


@dataclass
class Trial:
    path: Path
    date: str          # MMDDYYYY
    led_period_ms: int # nominal T_LED in ms
    index: int         # within-date trial counter
    ff_onsets: np.ndarray   # firefly flash onset times (s)
    led_onsets: np.ndarray  # LED flash onset times (s)
    duration_s: float       # total recording duration

    @property
    def habituation_end(self) -> float:
        """First LED-on time, or None if no LED pulses recorded."""
        return float(self.led_onsets[0]) if self.led_onsets.size else float("nan")

    @property
    def habituation_duration(self) -> float:
        """Time from first FF onset (proxy: t=0) to first LED onset."""
        return self.habituation_end

    @property
    def ff_habituation(self) -> np.ndarray:
        if self.led_onsets.size == 0:
            return self.ff_onsets
        return self.ff_onsets[self.ff_onsets < self.led_onsets[0]]

    @property
    def ff_perturbed(self) -> np.ndarray:
        if self.led_onsets.size == 0:
            return np.array([])
        return self.ff_onsets[self.ff_onsets >= self.led_onsets[0]]


def _parse_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ff_t, ff_s, led_t, led_s = [], [], [], []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                a = ast.literal_eval(row[0])
                b = ast.literal_eval(row[1])
            except (SyntaxError, ValueError):
                continue
            ff_t.append(a[0]); ff_s.append(a[1])
            led_t.append(b[0]); led_s.append(b[1])
    return (np.asarray(ff_t, dtype=float),
            np.asarray(ff_s, dtype=float),
            np.asarray(led_t, dtype=float),
            np.asarray(led_s, dtype=float))


def _rising_edges(t: np.ndarray, s: np.ndarray, low: float, high: float) -> np.ndarray:
    """Indices where state transitions low -> high."""
    if s.size < 2:
        return np.array([], dtype=int)
    mask = (s[1:] == high) & (s[:-1] == low)
    return np.where(mask)[0] + 1


def load_trial(path: Path) -> Trial | None:
    m = FILE_RE.match(path.name)
    if m is None:
        return None
    date, led_period_ms, index = m.group(1), int(m.group(2)), int(m.group(3))
    ff_t, ff_s, led_t, led_s = _parse_csv(path)
    if ff_t.size == 0:
        return None
    ff_onsets = ff_t[_rising_edges(ff_t, ff_s, low=0.0, high=1.0)]
    led_onsets = led_t[_rising_edges(led_t, led_s, low=1.0, high=2.0)]
    duration = float(max(ff_t[-1], led_t[-1]))
    return Trial(
        path=path, date=date, led_period_ms=led_period_ms, index=index,
        ff_onsets=ff_onsets, led_onsets=led_onsets, duration_s=duration,
    )


def iter_trials(led_periods: list[int] | None = None) -> Iterator[Trial]:
    for path in sorted(DATA_DIR.iterdir()):
        if not path.name.endswith(".csv"):
            continue
        trial = load_trial(path)
        if trial is None:
            continue
        if led_periods is not None and trial.led_period_ms not in led_periods:
            continue
        yield trial


# ---- shared signal-processing utilities ----

def bout_segments(onsets: np.ndarray, gap_s: float = 2.0, min_flashes: int = 3) -> list[np.ndarray]:
    """Split an onset sequence into bouts of consecutive flashes separated by <= gap_s.
    Keep bouts with at least min_flashes onsets."""
    if onsets.size == 0:
        return []
    splits = np.where(np.diff(onsets) > gap_s)[0] + 1
    pieces = np.split(onsets, splits)
    return [p for p in pieces if p.size >= min_flashes]


def perturbed_bouts(trial, gap_s: float = 2.0, min_flashes: int = 3) -> list[np.ndarray]:
    """Bouts of firefly flashes after the first LED pulse, mirroring the
    codebase's `get_bout_indices(max_gap=2.0)` followed by `bout_analysis`
    keeping bouts of length > 2.
    """
    return bout_segments(trial.ff_perturbed, gap_s=gap_s, min_flashes=min_flashes)


def prior_period(trial) -> float:
    """Median IFI of habituation flashes (codebase's `prior_period` from
    `windowed_rqa`). Returns NaN if fewer than 2 habituation flashes."""
    h = trial.ff_habituation
    if h.size < 2:
        return float("nan")
    return float(np.median(np.diff(h)))


# ---- canonical PRC pair generator (codebase's analyze_firefly_led_synchronization,
#      SELECTION_MODE="midpoint", n_bins=35) ----

PRC_MIN_ISI = 0.30          # seconds, matches codebase MIN_ISI
PRC_THRESH_FACTOR = 2.0     # T_next < 2 * T_LED  (codebase `thresh`)
PRC_BOUT_GAP = 2.0          # seconds
PRC_BOUT_MIN_FLASHES = 6    # bouts must have num_flashes = end_idx-start_idx >= 5
                            # (codebase `exclude = 5`), i.e. >= 6 actual flashes
PRC_LED_TOL = 1e-3          # 1 ms tolerance, matches codebase `tol`


def _wrap_half(x):
    """Codebase's _wrap_half: x - floor(x + 0.5).  Same as
    `(x + 0.5) mod 1 - 0.5` for real x — manuscript Eq. (4)."""
    return x - np.floor(x + 0.5)


def prc_pairs_midpoint(trial):
    """Yield (phi, R, T_prev, T_next, idx_prev) for each cycle that survives
    the codebase's SELECTION_MODE='midpoint' filters in
    `analyze_firefly_led_synchronization`.  ``idx_prev`` is the absolute index
    of ``t_prev`` in ``trial.ff_onsets`` (the *full* firefly stream including
    habituation), so callers can look back to true preceding IFIs.

      * iterate consecutive triples (t_prevprev, t_prev, t_next) of perturbed
        firefly onsets;
      * cycle (t_prev, t_next) must overlap a bout window (gap <= 2.0 s,
        bout flash-count diff >= 5);
      * T_prev > 0.30 s, T_next > 0.30 s, T_next < 2 * T_LED;
      * pick the LED inside (t_prev + 1ms, t_next - 1ms) nearest to the
        midpoint of (t_prev, t_next);
      * phi = wrap_half((L - t_prev) / T_prev);
      * R = T_next / T_prev.
    """
    ff_perturbed = trial.ff_perturbed
    ff_all = trial.ff_onsets
    led = trial.led_onsets
    if ff_perturbed.size < 3 or led.size == 0:
        return
    # offset from ff_perturbed indices to ff_all indices
    offset = int((ff_all < (led[0] if led.size else np.inf)).sum())
    T_LED = trial.led_period_ms / 1000.0
    bouts = bout_segments(ff_perturbed, gap_s=PRC_BOUT_GAP,
                          min_flashes=PRC_BOUT_MIN_FLASHES)
    if not bouts:
        return
    for bout in bouts:
        start, end = float(bout[0]), float(bout[-1])
        l_win = led[(led >= start) & (led <= end)]
        if l_win.size == 0:
            continue
        for j in range(1, ff_perturbed.size - 1):
            t_prevprev = ff_perturbed[j - 1]
            t_prev = ff_perturbed[j]
            t_next = ff_perturbed[j + 1]
            if (t_next <= start) or (t_prev >= end):
                continue
            T_prev = t_prev - t_prevprev
            T_next = t_next - t_prev
            if (T_prev < PRC_MIN_ISI) or (T_next < PRC_MIN_ISI):
                continue
            if T_next >= PRC_THRESH_FACTOR * T_LED:
                continue
            m = (l_win > t_prev + PRC_LED_TOL) & (l_win < t_next - PRC_LED_TOL)
            leds_inside = l_win[m]
            if leds_inside.size == 0:
                continue
            mid = 0.5 * (t_prev + t_next)
            L = leds_inside[np.argmin(np.abs(leds_inside - mid))]
            phi = float(_wrap_half((L - t_prev) / T_prev))
            if not (-0.5 <= phi < 0.5):
                continue
            yield (phi, float(T_next / T_prev),
                   float(T_prev), float(T_next), offset + j)


def interflash_intervals(onsets: np.ndarray) -> np.ndarray:
    return np.diff(onsets) if onsets.size >= 2 else np.array([])


def phi_eval_manuscript(phi_pre: np.ndarray | float) -> np.ndarray | float:
    """Manuscript Eq. (4):  phi_eval = (phi_pre + 1/2) mod 1 - 1/2.

    For phi_pre in (0, 1/2) returns phi_eval = phi_pre   (positive: LED in first
    half, ie. after the *prior* firefly flash).
    For phi_pre in (1/2, 1) returns phi_eval = phi_pre-1 (negative: LED in second
    half, ie. before the *next* firefly flash).
    """
    pp = np.asarray(phi_pre)
    return (pp + 0.5) % 1.0 - 0.5


def phi_eval_for_pulses(ff_prev: float, ff_next: float, led_pulses: np.ndarray) -> np.ndarray:
    """Signed phase of each LED pulse within (ff_prev, ff_next), per Eq. (4)."""
    if ff_next <= ff_prev:
        return np.array([])
    T = ff_next - ff_prev
    mask = (led_pulses > ff_prev) & (led_pulses < ff_next)
    phi_pre = (led_pulses[mask] - ff_prev) / T
    return phi_eval_manuscript(phi_pre)


def prc_records_midpoint(trial):
    """Same scope and selection rule as `prc_pairs_midpoint`, but yields a dict
    per in-scope cycle with the extra bookkeeping needed by the R2 analyses
    (B9--B11):

      trial        file stem (one trial = one individual firefly)
      led_period   nominal T_LED in ms
      phi_pre      unwrapped LED phase (L - t_prev) / T_prev  (can exceed 1)
      phi          wrapped phase, manuscript phi_eval = wrap_half(phi_pre)
      phi_next     alternative: wrap_half((L - t_prev) / T_next)
      R            T_next / T_prev
      n_prev       number of LED pulses strictly inside (t_prevprev, t_prev)
      n_next       number of LED pulses strictly inside (t_prev, t_next)
    """
    ff_all = trial.ff_onsets
    led = trial.led_onsets
    for phi, R, T_prev, T_next, idx_prev in prc_pairs_midpoint(trial):
        t_prevprev, t_prev, t_next = ff_all[idx_prev - 1], ff_all[idx_prev], ff_all[idx_prev + 1]
        inside = led[(led > t_prev + PRC_LED_TOL) & (led < t_next - PRC_LED_TOL)]
        mid = 0.5 * (t_prev + t_next)
        L = float(inside[np.argmin(np.abs(inside - mid))])
        yield dict(
            trial=trial.path.stem, led_period=trial.led_period_ms,
            phi_pre=(L - t_prev) / T_prev, phi=phi,
            phi_next=float(_wrap_half((L - t_prev) / T_next)),
            R=R, T_prev=T_prev, T_next=T_next,
            n_prev=int(((led > t_prevprev) & (led < t_prev)).sum()),
            n_next=int(inside.size),
        )


def all_prc_records():
    """DataFrame-ready list of `prc_records_midpoint` dicts over all trials."""
    out = []
    for trial in iter_trials():
        out.extend(prc_records_midpoint(trial))
    return out
