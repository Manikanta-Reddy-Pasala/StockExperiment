# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2024-09-03 00:00:00 → 2026-05-11 00:00:00 (416 bars)
- **Last close:** 4523.60
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 4
- **Avg / median % per leg:** 5.89% / 6.92%
- **Sum % (uncompounded):** 53.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 2 | 3 | 4 | 5.89% | 53.0% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 3 | 4 | 5.89% | 53.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 2 | 3 | 4 | 5.89% | 53.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 00:00:00 | 2417.10 | 1879.57 | 2338.48 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=83.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 00:00:00 | 2584.26 | 1929.93 | 2418.74 | T1 booked 50% @ 2584.26 |
| Target hit | 2025-08-26 00:00:00 | 2690.10 | 2072.50 | 2724.77 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 00:00:00 | 2924.60 | 2164.18 | 2771.86 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=97.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 00:00:00 | 3119.01 | 2197.65 | 2853.60 | T1 booked 50% @ 3119.01 |
| Stop hit — per-position SL triggered | 2025-09-26 00:00:00 | 2924.60 | 2227.92 | 2890.65 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2025-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 00:00:00 | 3170.10 | 2260.12 | 2945.35 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=101.46 |
| Stop hit — per-position SL triggered | 2025-10-09 00:00:00 | 3017.90 | 2292.57 | 2991.10 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2025-12-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 00:00:00 | 3097.80 | 2557.91 | 2951.24 | Stage2 pullback-breakout RSI=59 vol=3.4x ATR=136.98 |
| Stop hit — per-position SL triggered | 2026-01-06 00:00:00 | 3081.00 | 2610.74 | 3051.51 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 00:00:00 | 3157.90 | 2635.82 | 2849.98 | Stage2 pullback-breakout RSI=63 vol=3.5x ATR=139.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 00:00:00 | 3437.40 | 2662.10 | 3003.20 | T1 booked 50% @ 3437.40 |
| Target hit | 2026-03-13 00:00:00 | 3604.30 | 2908.68 | 3691.61 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 00:00:00 | 4067.00 | 3036.65 | 3752.65 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=214.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 00:00:00 | 4495.07 | 3126.87 | 4017.43 | T1 booked 50% @ 4495.07 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-17 00:00:00 | 2417.10 | 2025-07-30 00:00:00 | 2584.26 | PARTIAL | 0.50 | 6.92% |
| BUY | retest1 | 2025-07-17 00:00:00 | 2417.10 | 2025-08-26 00:00:00 | 2690.10 | TARGET_HIT | 0.50 | 11.29% |
| BUY | retest1 | 2025-09-16 00:00:00 | 2924.60 | 2025-09-22 00:00:00 | 3119.01 | PARTIAL | 0.50 | 6.65% |
| BUY | retest1 | 2025-09-16 00:00:00 | 2924.60 | 2025-09-26 00:00:00 | 2924.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-03 00:00:00 | 3170.10 | 2025-10-09 00:00:00 | 3017.90 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest1 | 2025-12-22 00:00:00 | 3097.80 | 2026-01-06 00:00:00 | 3081.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-01-29 00:00:00 | 3157.90 | 2026-02-03 00:00:00 | 3437.40 | PARTIAL | 0.50 | 8.85% |
| BUY | retest1 | 2026-01-29 00:00:00 | 3157.90 | 2026-03-13 00:00:00 | 3604.30 | TARGET_HIT | 0.50 | 14.14% |
| BUY | retest1 | 2026-04-10 00:00:00 | 4067.00 | 2026-04-23 00:00:00 | 4495.07 | PARTIAL | 0.50 | 10.53% |
