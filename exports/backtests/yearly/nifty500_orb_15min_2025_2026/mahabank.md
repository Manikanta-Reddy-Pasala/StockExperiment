# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 83.90
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
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 14 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 72
- **Target hits / Stop hits / Partials:** 14 / 72 / 39
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 23.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 28 | 43.8% | 8 | 36 | 20 | 0.26% | 16.4% |
| BUY @ 2nd Alert (retest1) | 64 | 28 | 43.8% | 8 | 36 | 20 | 0.26% | 16.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 25 | 41.0% | 6 | 36 | 19 | 0.12% | 7.5% |
| SELL @ 2nd Alert (retest1) | 61 | 25 | 41.0% | 6 | 36 | 19 | 0.12% | 7.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 125 | 53 | 42.4% | 14 | 72 | 39 | 0.19% | 23.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 50.89 | 50.64 | 0.00 | ORB-long ORB[50.32,50.74] vol=2.6x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 50.76 | 50.65 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 11:00:00 | 50.80 | 50.53 | 0.00 | ORB-long ORB[50.29,50.69] vol=2.4x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 12:40:00 | 50.97 | 50.62 | 0.00 | T1 1.5R @ 50.97 |
| Target hit | 2025-05-23 15:20:00 | 51.36 | 50.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 50.82 | 51.08 | 0.00 | ORB-short ORB[50.95,51.49] vol=2.1x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 09:50:00 | 50.62 | 50.98 | 0.00 | T1 1.5R @ 50.62 |
| Stop hit — per-position SL triggered | 2025-05-27 10:50:00 | 50.82 | 50.90 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 11:15:00 | 51.44 | 51.29 | 0.00 | ORB-long ORB[50.96,51.34] vol=4.8x ATR=0.11 |
| Stop hit — per-position SL triggered | 2025-05-28 11:20:00 | 51.33 | 51.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 51.12 | 51.32 | 0.00 | ORB-short ORB[51.17,51.59] vol=1.6x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-05-29 09:35:00 | 51.24 | 51.26 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:15:00 | 50.90 | 51.10 | 0.00 | ORB-short ORB[50.95,51.23] vol=1.7x ATR=0.09 |
| Stop hit — per-position SL triggered | 2025-05-30 10:25:00 | 50.99 | 51.07 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 55.45 | 55.92 | 0.00 | ORB-short ORB[55.77,56.44] vol=1.6x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:55:00 | 55.23 | 55.80 | 0.00 | T1 1.5R @ 55.23 |
| Target hit | 2025-06-12 15:20:00 | 54.07 | 54.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-06-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:50:00 | 53.42 | 53.93 | 0.00 | ORB-short ORB[53.95,54.60] vol=3.5x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-06-19 10:55:00 | 53.60 | 53.91 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:00:00 | 55.12 | 54.68 | 0.00 | ORB-long ORB[54.07,54.84] vol=3.5x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-06-24 10:05:00 | 54.91 | 54.70 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 11:05:00 | 54.47 | 54.67 | 0.00 | ORB-short ORB[54.56,54.98] vol=1.8x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-06-25 11:10:00 | 54.59 | 54.67 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:00:00 | 54.07 | 54.31 | 0.00 | ORB-short ORB[54.17,54.70] vol=1.8x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:15:00 | 53.87 | 54.21 | 0.00 | T1 1.5R @ 53.87 |
| Target hit | 2025-06-26 15:20:00 | 53.96 | 53.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:20:00 | 55.07 | 54.64 | 0.00 | ORB-long ORB[54.06,54.86] vol=1.7x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:25:00 | 55.37 | 54.72 | 0.00 | T1 1.5R @ 55.37 |
| Stop hit — per-position SL triggered | 2025-06-27 10:35:00 | 55.07 | 54.79 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 56.85 | 57.21 | 0.00 | ORB-short ORB[57.19,57.57] vol=2.5x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-07-08 09:45:00 | 57.01 | 57.17 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:15:00 | 56.10 | 56.37 | 0.00 | ORB-short ORB[56.42,56.77] vol=2.5x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:05:00 | 55.88 | 56.27 | 0.00 | T1 1.5R @ 55.88 |
| Stop hit — per-position SL triggered | 2025-07-11 12:35:00 | 56.10 | 56.15 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 56.48 | 56.10 | 0.00 | ORB-long ORB[55.61,56.35] vol=3.0x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-07-14 10:00:00 | 56.26 | 56.18 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:00:00 | 56.96 | 56.55 | 0.00 | ORB-long ORB[56.22,56.62] vol=4.0x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 10:10:00 | 57.24 | 56.67 | 0.00 | T1 1.5R @ 57.24 |
| Stop hit — per-position SL triggered | 2025-07-15 10:20:00 | 56.96 | 56.71 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:10:00 | 56.84 | 57.10 | 0.00 | ORB-short ORB[56.90,57.43] vol=1.7x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 56.65 | 57.06 | 0.00 | T1 1.5R @ 56.65 |
| Stop hit — per-position SL triggered | 2025-07-18 11:05:00 | 56.84 | 56.93 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:45:00 | 56.96 | 56.75 | 0.00 | ORB-long ORB[56.35,56.93] vol=1.9x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:25:00 | 57.19 | 56.84 | 0.00 | T1 1.5R @ 57.19 |
| Stop hit — per-position SL triggered | 2025-07-21 10:50:00 | 56.96 | 56.87 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 57.11 | 56.99 | 0.00 | ORB-long ORB[56.60,57.10] vol=2.7x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:40:00 | 57.31 | 57.03 | 0.00 | T1 1.5R @ 57.31 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 57.11 | 57.08 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:55:00 | 55.80 | 56.02 | 0.00 | ORB-short ORB[55.86,56.28] vol=2.3x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-08-01 10:30:00 | 55.92 | 55.96 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:00:00 | 54.78 | 54.97 | 0.00 | ORB-short ORB[54.90,55.34] vol=1.5x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 54.90 | 54.93 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:40:00 | 53.25 | 53.50 | 0.00 | ORB-short ORB[53.37,53.80] vol=1.9x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:30:00 | 53.04 | 53.38 | 0.00 | T1 1.5R @ 53.04 |
| Target hit | 2025-08-07 14:35:00 | 53.11 | 53.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — SELL (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 55.51 | 55.84 | 0.00 | ORB-short ORB[55.55,56.25] vol=2.4x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:40:00 | 55.29 | 55.78 | 0.00 | T1 1.5R @ 55.29 |
| Stop hit — per-position SL triggered | 2025-08-11 15:00:00 | 55.51 | 55.70 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:10:00 | 55.33 | 55.60 | 0.00 | ORB-short ORB[55.50,56.07] vol=4.1x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:45:00 | 55.17 | 55.55 | 0.00 | T1 1.5R @ 55.17 |
| Stop hit — per-position SL triggered | 2025-08-13 13:05:00 | 55.33 | 55.47 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 54.72 | 55.00 | 0.00 | ORB-short ORB[54.85,55.34] vol=1.7x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-08-14 09:40:00 | 54.84 | 54.93 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:30:00 | 55.85 | 55.49 | 0.00 | ORB-long ORB[55.06,55.54] vol=4.7x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-08-21 10:35:00 | 55.72 | 55.50 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 54.50 | 54.62 | 0.00 | ORB-short ORB[54.51,54.97] vol=1.5x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 10:15:00 | 54.32 | 54.51 | 0.00 | T1 1.5R @ 54.32 |
| Stop hit — per-position SL triggered | 2025-08-25 11:55:00 | 54.50 | 54.48 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 11:10:00 | 51.77 | 52.18 | 0.00 | ORB-short ORB[51.85,52.60] vol=2.2x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-09-01 13:25:00 | 51.90 | 52.05 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 11:15:00 | 52.48 | 52.84 | 0.00 | ORB-short ORB[52.80,53.35] vol=1.7x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 11:30:00 | 52.35 | 52.80 | 0.00 | T1 1.5R @ 52.35 |
| Target hit | 2025-09-04 15:20:00 | 52.02 | 52.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 51.93 | 52.12 | 0.00 | ORB-short ORB[52.04,52.43] vol=2.9x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 52.06 | 52.10 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:55:00 | 53.12 | 52.80 | 0.00 | ORB-long ORB[52.40,52.95] vol=2.5x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:05:00 | 53.37 | 52.88 | 0.00 | T1 1.5R @ 53.37 |
| Stop hit — per-position SL triggered | 2025-09-08 10:50:00 | 53.12 | 52.95 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 53.05 | 52.90 | 0.00 | ORB-long ORB[52.56,52.94] vol=3.4x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 52.92 | 52.91 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:30:00 | 54.93 | 54.65 | 0.00 | ORB-long ORB[54.20,54.88] vol=3.0x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 09:35:00 | 55.22 | 54.75 | 0.00 | T1 1.5R @ 55.22 |
| Stop hit — per-position SL triggered | 2025-09-11 09:40:00 | 54.93 | 54.79 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:00:00 | 53.90 | 54.10 | 0.00 | ORB-short ORB[53.91,54.58] vol=1.8x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:15:00 | 53.74 | 54.05 | 0.00 | T1 1.5R @ 53.74 |
| Stop hit — per-position SL triggered | 2025-09-12 13:25:00 | 53.90 | 53.90 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:50:00 | 55.64 | 55.36 | 0.00 | ORB-long ORB[55.10,55.45] vol=4.1x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 09:55:00 | 55.85 | 55.60 | 0.00 | T1 1.5R @ 55.85 |
| Target hit | 2025-09-17 15:20:00 | 57.29 | 56.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:40:00 | 57.50 | 57.25 | 0.00 | ORB-long ORB[56.87,57.45] vol=1.9x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 09:50:00 | 57.80 | 57.38 | 0.00 | T1 1.5R @ 57.80 |
| Stop hit — per-position SL triggered | 2025-09-19 10:00:00 | 57.50 | 57.44 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:55:00 | 56.44 | 56.70 | 0.00 | ORB-short ORB[56.75,57.29] vol=1.6x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-09-23 10:00:00 | 56.60 | 56.70 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:05:00 | 58.16 | 57.63 | 0.00 | ORB-long ORB[57.22,57.99] vol=5.6x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:10:00 | 58.49 | 57.81 | 0.00 | T1 1.5R @ 58.49 |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 58.16 | 57.85 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:40:00 | 54.70 | 54.34 | 0.00 | ORB-long ORB[53.91,54.42] vol=2.1x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:00:00 | 54.97 | 54.58 | 0.00 | T1 1.5R @ 54.97 |
| Target hit | 2025-09-29 11:25:00 | 54.91 | 54.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2025-10-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:40:00 | 56.97 | 57.09 | 0.00 | ORB-short ORB[57.05,57.36] vol=3.6x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 09:50:00 | 56.79 | 57.00 | 0.00 | T1 1.5R @ 56.79 |
| Stop hit — per-position SL triggered | 2025-10-08 10:05:00 | 56.97 | 56.98 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 56.99 | 56.84 | 0.00 | ORB-long ORB[56.61,56.97] vol=2.1x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-10-10 09:35:00 | 56.86 | 56.86 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 57.28 | 57.60 | 0.00 | ORB-short ORB[57.45,58.00] vol=1.6x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 09:35:00 | 57.00 | 57.56 | 0.00 | T1 1.5R @ 57.00 |
| Stop hit — per-position SL triggered | 2025-10-13 09:40:00 | 57.28 | 57.55 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:10:00 | 57.80 | 57.24 | 0.00 | ORB-long ORB[57.06,57.60] vol=3.0x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-10-17 11:20:00 | 57.60 | 57.27 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:35:00 | 58.60 | 57.98 | 0.00 | ORB-long ORB[57.36,58.23] vol=2.9x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 09:40:00 | 58.98 | 58.24 | 0.00 | T1 1.5R @ 58.98 |
| Stop hit — per-position SL triggered | 2025-10-20 09:45:00 | 58.60 | 58.26 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:15:00 | 60.84 | 60.27 | 0.00 | ORB-long ORB[59.87,60.50] vol=2.9x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-10-23 10:20:00 | 60.58 | 60.28 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:55:00 | 58.88 | 59.38 | 0.00 | ORB-short ORB[59.34,59.75] vol=2.3x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 59.06 | 59.36 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 58.55 | 58.75 | 0.00 | ORB-short ORB[58.62,59.07] vol=2.1x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 09:40:00 | 58.30 | 58.65 | 0.00 | T1 1.5R @ 58.30 |
| Stop hit — per-position SL triggered | 2025-10-27 10:20:00 | 58.55 | 58.53 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:30:00 | 59.30 | 59.58 | 0.00 | ORB-short ORB[59.42,60.00] vol=1.6x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 09:55:00 | 59.07 | 59.41 | 0.00 | T1 1.5R @ 59.07 |
| Stop hit — per-position SL triggered | 2025-10-30 10:20:00 | 59.30 | 59.35 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:10:00 | 59.22 | 59.48 | 0.00 | ORB-short ORB[59.35,59.92] vol=2.1x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:15:00 | 58.96 | 59.42 | 0.00 | T1 1.5R @ 58.96 |
| Target hit | 2025-11-04 15:20:00 | 58.51 | 58.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2025-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:40:00 | 58.25 | 58.51 | 0.00 | ORB-short ORB[58.41,59.10] vol=1.7x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-11-06 10:45:00 | 58.44 | 58.50 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:45:00 | 57.95 | 57.36 | 0.00 | ORB-long ORB[57.11,57.58] vol=2.4x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-11-07 10:55:00 | 57.73 | 57.39 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:30:00 | 57.34 | 57.46 | 0.00 | ORB-short ORB[57.36,57.65] vol=2.0x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:35:00 | 57.18 | 57.42 | 0.00 | T1 1.5R @ 57.18 |
| Stop hit — per-position SL triggered | 2025-11-13 09:40:00 | 57.34 | 57.41 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 57.59 | 57.34 | 0.00 | ORB-long ORB[56.75,57.45] vol=1.9x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-11-14 10:00:00 | 57.42 | 57.54 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:05:00 | 58.46 | 59.05 | 0.00 | ORB-short ORB[59.13,59.78] vol=3.2x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-11-21 11:20:00 | 58.63 | 58.96 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 60.73 | 60.20 | 0.00 | ORB-long ORB[59.63,60.15] vol=3.5x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-11-26 09:35:00 | 60.52 | 60.24 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:35:00 | 58.89 | 59.15 | 0.00 | ORB-short ORB[59.06,59.53] vol=3.3x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:00:00 | 58.66 | 59.02 | 0.00 | T1 1.5R @ 58.66 |
| Stop hit — per-position SL triggered | 2025-11-28 10:50:00 | 58.89 | 58.88 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:20:00 | 58.48 | 59.17 | 0.00 | ORB-short ORB[58.80,59.55] vol=3.0x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-12-01 10:30:00 | 58.77 | 59.02 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:00:00 | 57.14 | 56.86 | 0.00 | ORB-long ORB[56.60,56.99] vol=1.6x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-12-05 10:05:00 | 57.02 | 56.90 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:15:00 | 56.23 | 55.80 | 0.00 | ORB-long ORB[55.51,56.14] vol=2.3x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:30:00 | 56.55 | 55.93 | 0.00 | T1 1.5R @ 56.55 |
| Stop hit — per-position SL triggered | 2025-12-09 10:35:00 | 56.23 | 55.94 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:35:00 | 57.59 | 57.28 | 0.00 | ORB-long ORB[56.89,57.52] vol=3.4x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-12-10 09:45:00 | 57.40 | 57.31 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:30:00 | 57.50 | 57.05 | 0.00 | ORB-long ORB[56.55,56.99] vol=3.8x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-12-11 10:40:00 | 57.33 | 57.20 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:40:00 | 56.97 | 57.29 | 0.00 | ORB-short ORB[57.26,57.70] vol=1.6x ATR=0.14 |
| Stop hit — per-position SL triggered | 2025-12-12 11:20:00 | 57.11 | 57.25 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:55:00 | 56.82 | 57.03 | 0.00 | ORB-short ORB[56.93,57.53] vol=2.0x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-12-16 10:35:00 | 56.94 | 56.96 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:45:00 | 56.30 | 56.40 | 0.00 | ORB-short ORB[56.48,56.80] vol=1.6x ATR=0.14 |
| Stop hit — per-position SL triggered | 2025-12-18 12:10:00 | 56.44 | 56.37 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:55:00 | 57.65 | 57.37 | 0.00 | ORB-long ORB[57.02,57.40] vol=2.5x ATR=0.14 |
| Stop hit — per-position SL triggered | 2025-12-22 10:20:00 | 57.51 | 57.48 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:15:00 | 57.98 | 57.82 | 0.00 | ORB-long ORB[57.56,57.94] vol=2.5x ATR=0.14 |
| Stop hit — per-position SL triggered | 2025-12-23 11:25:00 | 57.84 | 57.88 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 57.59 | 57.61 | 0.00 | ORB-short ORB[57.62,57.89] vol=2.2x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-12-24 11:25:00 | 57.72 | 57.62 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:35:00 | 57.47 | 57.31 | 0.00 | ORB-long ORB[57.10,57.34] vol=2.1x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-12-26 10:50:00 | 57.34 | 57.32 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:15:00 | 57.98 | 57.65 | 0.00 | ORB-long ORB[57.29,57.85] vol=2.4x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:20:00 | 58.20 | 57.81 | 0.00 | T1 1.5R @ 58.20 |
| Target hit | 2025-12-30 15:20:00 | 60.82 | 59.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2026-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:00:00 | 62.90 | 62.27 | 0.00 | ORB-long ORB[61.70,62.50] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-01-01 10:15:00 | 62.65 | 62.37 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:55:00 | 64.70 | 64.19 | 0.00 | ORB-long ORB[63.83,64.50] vol=2.3x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:05:00 | 65.03 | 64.47 | 0.00 | T1 1.5R @ 65.03 |
| Stop hit — per-position SL triggered | 2026-01-06 10:10:00 | 64.70 | 64.48 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 62.81 | 63.08 | 0.00 | ORB-short ORB[62.90,63.49] vol=2.2x ATR=0.16 |
| Stop hit — per-position SL triggered | 2026-01-08 09:35:00 | 62.97 | 63.07 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:40:00 | 63.70 | 62.84 | 0.00 | ORB-long ORB[61.99,62.78] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-01-09 10:55:00 | 63.45 | 62.93 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 09:35:00 | 63.05 | 62.66 | 0.00 | ORB-long ORB[62.18,63.00] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-01-12 09:45:00 | 62.76 | 62.72 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-01-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:00:00 | 66.60 | 66.20 | 0.00 | ORB-long ORB[65.40,66.27] vol=2.4x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:05:00 | 66.95 | 66.31 | 0.00 | T1 1.5R @ 66.95 |
| Target hit | 2026-01-23 12:30:00 | 66.77 | 66.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 76 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 66.37 | 66.13 | 0.00 | ORB-long ORB[65.73,66.29] vol=2.2x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:55:00 | 66.65 | 66.27 | 0.00 | T1 1.5R @ 66.65 |
| Target hit | 2026-02-12 12:00:00 | 66.55 | 66.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 77 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 66.23 | 65.91 | 0.00 | ORB-long ORB[65.44,65.98] vol=2.8x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:25:00 | 66.52 | 66.02 | 0.00 | T1 1.5R @ 66.52 |
| Target hit | 2026-02-17 15:20:00 | 67.35 | 67.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 68.93 | 69.06 | 0.00 | ORB-short ORB[68.97,69.85] vol=1.5x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:40:00 | 68.63 | 69.05 | 0.00 | T1 1.5R @ 68.63 |
| Target hit | 2026-02-19 15:20:00 | 67.70 | 68.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 73.95 | 72.83 | 0.00 | ORB-long ORB[71.68,72.76] vol=4.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:00:00 | 74.53 | 73.26 | 0.00 | T1 1.5R @ 74.53 |
| Stop hit — per-position SL triggered | 2026-02-25 10:20:00 | 73.95 | 73.58 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:45:00 | 66.69 | 66.02 | 0.00 | ORB-long ORB[65.45,66.24] vol=1.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-03-12 09:50:00 | 66.37 | 66.06 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 70.44 | 70.69 | 0.00 | ORB-short ORB[70.45,71.45] vol=1.9x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 70.68 | 70.68 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 70.96 | 71.53 | 0.00 | ORB-short ORB[71.25,72.24] vol=1.6x ATR=0.21 |
| Stop hit — per-position SL triggered | 2026-04-15 11:50:00 | 71.17 | 71.47 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 72.09 | 71.64 | 0.00 | ORB-long ORB[71.28,71.99] vol=3.0x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-04-16 10:20:00 | 71.85 | 71.67 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 79.91 | 79.48 | 0.00 | ORB-long ORB[78.93,79.80] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 79.60 | 79.54 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 79.22 | 78.65 | 0.00 | ORB-long ORB[78.13,78.90] vol=1.6x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:00:00 | 79.70 | 78.88 | 0.00 | T1 1.5R @ 79.70 |
| Target hit | 2026-05-05 15:20:00 | 81.25 | 80.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 83.85 | 84.65 | 0.00 | ORB-short ORB[84.40,85.31] vol=1.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 84.16 | 84.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:30:00 | 50.89 | 2025-05-15 09:35:00 | 50.76 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-23 11:00:00 | 50.80 | 2025-05-23 12:40:00 | 50.97 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-05-23 11:00:00 | 50.80 | 2025-05-23 15:20:00 | 51.36 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2025-05-27 09:30:00 | 50.82 | 2025-05-27 09:50:00 | 50.62 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-27 09:30:00 | 50.82 | 2025-05-27 10:50:00 | 50.82 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 11:15:00 | 51.44 | 2025-05-28 11:20:00 | 51.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-29 09:30:00 | 51.12 | 2025-05-29 09:35:00 | 51.24 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-30 10:15:00 | 50.90 | 2025-05-30 10:25:00 | 50.99 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-06-12 11:10:00 | 55.45 | 2025-06-12 11:55:00 | 55.23 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-06-12 11:10:00 | 55.45 | 2025-06-12 15:20:00 | 54.07 | TARGET_HIT | 0.50 | 2.49% |
| SELL | retest1 | 2025-06-19 10:50:00 | 53.42 | 2025-06-19 10:55:00 | 53.60 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-24 10:00:00 | 55.12 | 2025-06-24 10:05:00 | 54.91 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-06-25 11:05:00 | 54.47 | 2025-06-25 11:10:00 | 54.59 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-26 10:00:00 | 54.07 | 2025-06-26 11:15:00 | 53.87 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-26 10:00:00 | 54.07 | 2025-06-26 15:20:00 | 53.96 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-06-27 10:20:00 | 55.07 | 2025-06-27 10:25:00 | 55.37 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-27 10:20:00 | 55.07 | 2025-06-27 10:35:00 | 55.07 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 09:35:00 | 56.85 | 2025-07-08 09:45:00 | 57.01 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-11 10:15:00 | 56.10 | 2025-07-11 11:05:00 | 55.88 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-11 10:15:00 | 56.10 | 2025-07-11 12:35:00 | 56.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 09:40:00 | 56.48 | 2025-07-14 10:00:00 | 56.26 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-07-15 10:00:00 | 56.96 | 2025-07-15 10:10:00 | 57.24 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-15 10:00:00 | 56.96 | 2025-07-15 10:20:00 | 56.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:10:00 | 56.84 | 2025-07-18 10:15:00 | 56.65 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-18 10:10:00 | 56.84 | 2025-07-18 11:05:00 | 56.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 09:45:00 | 56.96 | 2025-07-21 10:25:00 | 57.19 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-07-21 09:45:00 | 56.96 | 2025-07-21 10:50:00 | 56.96 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-24 09:35:00 | 57.11 | 2025-07-24 09:40:00 | 57.31 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-24 09:35:00 | 57.11 | 2025-07-24 10:15:00 | 57.11 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-01 09:55:00 | 55.80 | 2025-08-01 10:30:00 | 55.92 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-06 10:00:00 | 54.78 | 2025-08-06 10:20:00 | 54.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-07 10:40:00 | 53.25 | 2025-08-07 11:30:00 | 53.04 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-07 10:40:00 | 53.25 | 2025-08-07 14:35:00 | 53.11 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-08-11 11:10:00 | 55.51 | 2025-08-11 11:40:00 | 55.29 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-08-11 11:10:00 | 55.51 | 2025-08-11 15:00:00 | 55.51 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-13 11:10:00 | 55.33 | 2025-08-13 11:45:00 | 55.17 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-08-13 11:10:00 | 55.33 | 2025-08-13 13:05:00 | 55.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 09:30:00 | 54.72 | 2025-08-14 09:40:00 | 54.84 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-21 10:30:00 | 55.85 | 2025-08-21 10:35:00 | 55.72 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-25 09:30:00 | 54.50 | 2025-08-25 10:15:00 | 54.32 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-08-25 09:30:00 | 54.50 | 2025-08-25 11:55:00 | 54.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-01 11:10:00 | 51.77 | 2025-09-01 13:25:00 | 51.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-04 11:15:00 | 52.48 | 2025-09-04 11:30:00 | 52.35 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-09-04 11:15:00 | 52.48 | 2025-09-04 15:20:00 | 52.02 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-09-05 10:10:00 | 51.93 | 2025-09-05 10:20:00 | 52.06 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-08 09:55:00 | 53.12 | 2025-09-08 10:05:00 | 53.37 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-09-08 09:55:00 | 53.12 | 2025-09-08 10:50:00 | 53.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-10 09:40:00 | 53.05 | 2025-09-10 09:45:00 | 52.92 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-11 09:30:00 | 54.93 | 2025-09-11 09:35:00 | 55.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-09-11 09:30:00 | 54.93 | 2025-09-11 09:40:00 | 54.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 11:00:00 | 53.90 | 2025-09-12 11:15:00 | 53.74 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-09-12 11:00:00 | 53.90 | 2025-09-12 13:25:00 | 53.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 09:50:00 | 55.64 | 2025-09-17 09:55:00 | 55.85 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-17 09:50:00 | 55.64 | 2025-09-17 15:20:00 | 57.29 | TARGET_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2025-09-19 09:40:00 | 57.50 | 2025-09-19 09:50:00 | 57.80 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-09-19 09:40:00 | 57.50 | 2025-09-19 10:00:00 | 57.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 09:55:00 | 56.44 | 2025-09-23 10:00:00 | 56.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-24 10:05:00 | 58.16 | 2025-09-24 10:10:00 | 58.49 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-09-24 10:05:00 | 58.16 | 2025-09-24 10:15:00 | 58.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-29 09:40:00 | 54.70 | 2025-09-29 10:00:00 | 54.97 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-09-29 09:40:00 | 54.70 | 2025-09-29 11:25:00 | 54.91 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-08 09:40:00 | 56.97 | 2025-10-08 09:50:00 | 56.79 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-08 09:40:00 | 56.97 | 2025-10-08 10:05:00 | 56.97 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:30:00 | 56.99 | 2025-10-10 09:35:00 | 56.86 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-13 09:30:00 | 57.28 | 2025-10-13 09:35:00 | 57.00 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-10-13 09:30:00 | 57.28 | 2025-10-13 09:40:00 | 57.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 11:10:00 | 57.80 | 2025-10-17 11:20:00 | 57.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-20 09:35:00 | 58.60 | 2025-10-20 09:40:00 | 58.98 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-10-20 09:35:00 | 58.60 | 2025-10-20 09:45:00 | 58.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-23 10:15:00 | 60.84 | 2025-10-23 10:20:00 | 60.58 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-10-24 10:55:00 | 58.88 | 2025-10-24 11:15:00 | 59.06 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-27 09:30:00 | 58.55 | 2025-10-27 09:40:00 | 58.30 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-10-27 09:30:00 | 58.55 | 2025-10-27 10:20:00 | 58.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 09:30:00 | 59.30 | 2025-10-30 09:55:00 | 59.07 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-30 09:30:00 | 59.30 | 2025-10-30 10:20:00 | 59.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 10:10:00 | 59.22 | 2025-11-04 10:15:00 | 58.96 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-11-04 10:10:00 | 59.22 | 2025-11-04 15:20:00 | 58.51 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2025-11-06 10:40:00 | 58.25 | 2025-11-06 10:45:00 | 58.44 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-07 10:45:00 | 57.95 | 2025-11-07 10:55:00 | 57.73 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-13 09:30:00 | 57.34 | 2025-11-13 09:35:00 | 57.18 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-13 09:30:00 | 57.34 | 2025-11-13 09:40:00 | 57.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 09:30:00 | 57.59 | 2025-11-14 10:00:00 | 57.42 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-21 11:05:00 | 58.46 | 2025-11-21 11:20:00 | 58.63 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-26 09:30:00 | 60.73 | 2025-11-26 09:35:00 | 60.52 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-28 09:35:00 | 58.89 | 2025-11-28 10:00:00 | 58.66 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-28 09:35:00 | 58.89 | 2025-11-28 10:50:00 | 58.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 10:20:00 | 58.48 | 2025-12-01 10:30:00 | 58.77 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-12-05 10:00:00 | 57.14 | 2025-12-05 10:05:00 | 57.02 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-09 10:15:00 | 56.23 | 2025-12-09 10:30:00 | 56.55 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-12-09 10:15:00 | 56.23 | 2025-12-09 10:35:00 | 56.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 09:35:00 | 57.59 | 2025-12-10 09:45:00 | 57.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-11 10:30:00 | 57.50 | 2025-12-11 10:40:00 | 57.33 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-12 10:40:00 | 56.97 | 2025-12-12 11:20:00 | 57.11 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-16 09:55:00 | 56.82 | 2025-12-16 10:35:00 | 56.94 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-18 10:45:00 | 56.30 | 2025-12-18 12:10:00 | 56.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-22 09:55:00 | 57.65 | 2025-12-22 10:20:00 | 57.51 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-23 10:15:00 | 57.98 | 2025-12-23 11:25:00 | 57.84 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-24 10:55:00 | 57.59 | 2025-12-24 11:25:00 | 57.72 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-26 10:35:00 | 57.47 | 2025-12-26 10:50:00 | 57.34 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-30 10:15:00 | 57.98 | 2025-12-30 10:20:00 | 58.20 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-30 10:15:00 | 57.98 | 2025-12-30 15:20:00 | 60.82 | TARGET_HIT | 0.50 | 4.90% |
| BUY | retest1 | 2026-01-01 10:00:00 | 62.90 | 2026-01-01 10:15:00 | 62.65 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-01-06 09:55:00 | 64.70 | 2026-01-06 10:05:00 | 65.03 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-01-06 09:55:00 | 64.70 | 2026-01-06 10:10:00 | 64.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 09:30:00 | 62.81 | 2026-01-08 09:35:00 | 62.97 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-09 10:40:00 | 63.70 | 2026-01-09 10:55:00 | 63.45 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-12 09:35:00 | 63.05 | 2026-01-12 09:45:00 | 62.76 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-01-23 10:00:00 | 66.60 | 2026-01-23 10:05:00 | 66.95 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-01-23 10:00:00 | 66.60 | 2026-01-23 12:30:00 | 66.77 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-02-12 09:35:00 | 66.37 | 2026-02-12 09:55:00 | 66.65 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-12 09:35:00 | 66.37 | 2026-02-12 12:00:00 | 66.55 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-17 10:20:00 | 66.23 | 2026-02-17 10:25:00 | 66.52 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 10:20:00 | 66.23 | 2026-02-17 15:20:00 | 67.35 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2026-02-19 11:05:00 | 68.93 | 2026-02-19 11:40:00 | 68.63 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-19 11:05:00 | 68.93 | 2026-02-19 15:20:00 | 67.70 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2026-02-25 09:50:00 | 73.95 | 2026-02-25 10:00:00 | 74.53 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-02-25 09:50:00 | 73.95 | 2026-02-25 10:20:00 | 73.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 09:45:00 | 66.69 | 2026-03-12 09:50:00 | 66.37 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-04-10 10:05:00 | 70.44 | 2026-04-10 10:15:00 | 70.68 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-15 11:05:00 | 70.96 | 2026-04-15 11:50:00 | 71.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-16 10:15:00 | 72.09 | 2026-04-16 10:20:00 | 71.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-04 09:35:00 | 79.91 | 2026-05-04 09:50:00 | 79.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-05 09:45:00 | 79.22 | 2026-05-05 10:00:00 | 79.70 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-05-05 09:45:00 | 79.22 | 2026-05-05 15:20:00 | 81.25 | TARGET_HIT | 0.50 | 2.56% |
| SELL | retest1 | 2026-05-08 09:40:00 | 83.85 | 2026-05-08 10:10:00 | 84.16 | STOP_HIT | 1.00 | -0.37% |
