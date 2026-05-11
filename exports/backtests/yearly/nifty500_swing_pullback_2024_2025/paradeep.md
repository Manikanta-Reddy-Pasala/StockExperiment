# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 124.88
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 2
- **Avg / median % per leg:** -1.27% / -5.33%
- **Sum % (uncompounded):** -13.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | -1.27% | -14.0% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | -1.27% | -14.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 3 | 27.3% | 1 | 8 | 2 | -1.27% | -14.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 05:30:00 | 94.49 | 75.17 | 86.14 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=4.58 |
| Stop hit — per-position SL triggered | 2024-08-02 05:30:00 | 87.62 | 75.61 | 87.11 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-10-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 05:30:00 | 89.63 | 78.89 | 85.21 | Stage2 pullback-breakout RSI=60 vol=3.0x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-10-04 05:30:00 | 84.85 | 79.02 | 85.19 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-10-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 05:30:00 | 91.83 | 79.41 | 86.10 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=3.66 |
| Stop hit — per-position SL triggered | 2024-10-22 05:30:00 | 86.34 | 80.21 | 88.60 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-10-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 05:30:00 | 95.48 | 80.69 | 89.28 | Stage2 pullback-breakout RSI=60 vol=5.1x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 05:30:00 | 105.13 | 81.14 | 91.82 | T1 booked 50% @ 105.13 |
| Target hit | 2024-11-28 05:30:00 | 103.44 | 85.30 | 103.57 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-11-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 05:30:00 | 111.08 | 85.55 | 104.29 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-12-13 05:30:00 | 107.38 | 87.79 | 107.25 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-12-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 05:30:00 | 113.13 | 88.28 | 108.20 | Stage2 pullback-breakout RSI=61 vol=2.5x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-12-23 05:30:00 | 105.73 | 89.14 | 108.81 | SL hit (bars_held=4) |

### Cycle 7 — BUY (started 2025-01-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 05:30:00 | 120.66 | 90.97 | 111.47 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=4.83 |
| Stop hit — per-position SL triggered | 2025-01-10 05:30:00 | 113.41 | 92.28 | 113.71 | SL hit (bars_held=5) |

### Cycle 8 — BUY (started 2025-01-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 05:30:00 | 124.52 | 93.69 | 113.93 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=6.09 |
| Stop hit — per-position SL triggered | 2025-01-27 05:30:00 | 115.38 | 94.71 | 115.71 | SL hit (bars_held=4) |

### Cycle 9 — BUY (started 2025-03-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 05:30:00 | 103.54 | 95.70 | 94.56 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 05:30:00 | 113.25 | 96.47 | 101.43 | T1 booked 50% @ 113.25 |
| Stop hit — per-position SL triggered | 2025-04-07 05:30:00 | 103.54 | 96.87 | 104.18 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-30 05:30:00 | 94.49 | 2024-08-02 05:30:00 | 87.62 | STOP_HIT | 1.00 | -7.27% |
| BUY | retest1 | 2024-10-01 05:30:00 | 89.63 | 2024-10-04 05:30:00 | 84.85 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest1 | 2024-10-11 05:30:00 | 91.83 | 2024-10-22 05:30:00 | 86.34 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest1 | 2024-10-29 05:30:00 | 95.48 | 2024-10-31 05:30:00 | 105.13 | PARTIAL | 0.50 | 10.10% |
| BUY | retest1 | 2024-10-29 05:30:00 | 95.48 | 2024-11-28 05:30:00 | 103.44 | TARGET_HIT | 0.50 | 8.34% |
| BUY | retest1 | 2024-11-29 05:30:00 | 111.08 | 2024-12-13 05:30:00 | 107.38 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest1 | 2024-12-17 05:30:00 | 113.13 | 2024-12-23 05:30:00 | 105.73 | STOP_HIT | 1.00 | -6.54% |
| BUY | retest1 | 2025-01-03 05:30:00 | 120.66 | 2025-01-10 05:30:00 | 113.41 | STOP_HIT | 1.00 | -6.01% |
| BUY | retest1 | 2025-01-21 05:30:00 | 124.52 | 2025-01-27 05:30:00 | 115.38 | STOP_HIT | 1.00 | -7.34% |
| BUY | retest1 | 2025-03-24 05:30:00 | 103.54 | 2025-04-03 05:30:00 | 113.25 | PARTIAL | 0.50 | 9.38% |
| BUY | retest1 | 2025-03-24 05:30:00 | 103.54 | 2025-04-07 05:30:00 | 103.54 | STOP_HIT | 0.50 | 0.00% |
