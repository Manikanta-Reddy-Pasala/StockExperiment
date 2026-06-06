# breakout_swing_n500 — ARCHIVED (NO EDGE / ceiling too low)

**Status:** Archived 2026-06-06. NOT shipped, never wired into scheduler/registries.
**Spec:** `docs/superpowers/specs/2026-06-06-breakout-swing-design.md`

## What it was

Short-hold (1–5 trading-day) momentum-burst swing over the liquid N500. Two entry
alphas tried: fresh-40d-high breakout (vol surge + >SMA) and Medium-article
pullback-to-EMA. Exit via shared `breakout_exit_reason` (TARGET/STOP/TRAIL/MAX_HOLD).
No-lookahead (rank + transact same observed close). Reused shared breakout core.

## Why archived — exhaustive backtest, full-cycle 2021-03→2026-05, real fyers data

Three sweeps run (numpy, ~1700 configs total):

| Sweep | Configs | Best result | Kill-gate (Calmar≥2 + all-yrs-positive) |
|-------|--------:|-------------|:---:|
| Exit grid (orig)   | 972 | BO +10.8% CAGR / Calmar 0.36 | 0 |
| Structural          | 228 | BO +17.0% CAGR / Calmar 0.73 | 0 |
| Stage-B (tight/strong) | 432 | BO +16.1% / Calmar 0.88 | 0 |

**Ceiling = ~16-17% CAGR, ~60% WR, Calmar ~0.8.** No config clears the gate.

### Levers that mattered (the search WAS productive — just to a low ceiling)
- **Tight universe wins:** top-50 ADV leaders >> top-100/150 (bigger = noise).
- **Strong surge wins:** vol ≥ 4× 20d-avg >> 2×/3×. But ≥5×/6× and univ<50 HURT
  (too few trades, miss winners) → plateau at u50/v4.
- Momentum-rank ≈ vol-rank; HH window (20/40/60), SMA (100/200), mom-confirm = minor.
- Default loose config = −58% CAGR. Right levers → +17%. The edge is real but small.
- Pullback (Medium idea) strictly WORSE than breakout (+2% vs +17%).

### Why not shipped
- Calmar 0.8 << gate 2.0; 2026 (partial) negative.
- Strictly dominated by the live `emerging_momentum` rotation: **+114% CAGR / Calmar 3.0.**
  Adding this would dilute the portfolio.

## Lesson (consistent with ORB + midcap-on-PIT)

This stack's edge lives in **multi-day momentum ROTATION**, not fast/short-hold
trading. HFT (latency), intraday ORB (no edge), and now short-hold swing
(Calmar 0.8) all fail. Stop hunting sub-weekly alpha here.

Code kept for research. Reproduce: `backtest.py --sweep | --struct | --stageb`.
