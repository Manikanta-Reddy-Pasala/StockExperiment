# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 2968.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -0.58% / 0.00%
- **Sum % (uncompounded):** -3.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.58% | -3.5% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.58% | -3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.58% | -3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 2780.90 | 2610.69 | 2678.88 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=57.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 05:30:00 | 2895.26 | 2618.02 | 2725.77 | T1 booked 50% @ 2895.26 |
| Stop hit — per-position SL triggered | 2025-07-07 05:30:00 | 2780.90 | 2630.01 | 2768.33 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 2845.90 | 2661.73 | 2759.16 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=56.61 |
| Stop hit — per-position SL triggered | 2025-09-02 05:30:00 | 2779.80 | 2676.15 | 2789.24 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 2923.90 | 2719.83 | 2833.66 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=50.91 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 2847.53 | 2731.11 | 2854.22 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 2836.70 | 2735.52 | 2762.72 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=44.18 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 2842.40 | 2743.33 | 2798.88 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-01-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 05:30:00 | 2856.20 | 2753.96 | 2795.12 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=54.65 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 2774.22 | 2756.02 | 2797.66 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 2780.90 | 2025-06-27 05:30:00 | 2895.26 | PARTIAL | 0.50 | 4.11% |
| BUY | retest1 | 2025-06-24 05:30:00 | 2780.90 | 2025-07-07 05:30:00 | 2780.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-18 05:30:00 | 2845.90 | 2025-09-02 05:30:00 | 2779.80 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest1 | 2025-10-27 05:30:00 | 2923.90 | 2025-11-06 05:30:00 | 2847.53 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest1 | 2025-12-12 05:30:00 | 2836.70 | 2025-12-29 05:30:00 | 2842.40 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest1 | 2026-01-27 05:30:00 | 2856.20 | 2026-02-01 05:30:00 | 2774.22 | STOP_HIT | 1.00 | -2.87% |
