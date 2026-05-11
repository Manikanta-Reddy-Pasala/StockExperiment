# NHPC Ltd. (NHPC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 80.70
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
| ENTRY1 | 52 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 13 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 39
- **Target hits / Stop hits / Partials:** 13 / 39 / 24
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 18.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 12 | 33.3% | 3 | 24 | 9 | 0.14% | 4.9% |
| BUY @ 2nd Alert (retest1) | 36 | 12 | 33.3% | 3 | 24 | 9 | 0.14% | 4.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 40 | 25 | 62.5% | 10 | 15 | 15 | 0.34% | 13.6% |
| SELL @ 2nd Alert (retest1) | 40 | 25 | 62.5% | 10 | 15 | 15 | 0.34% | 13.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 76 | 37 | 48.7% | 13 | 39 | 24 | 0.24% | 18.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 104.70 | 105.45 | 0.00 | ORB-short ORB[105.05,106.30] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-05-24 09:40:00 | 105.18 | 105.38 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 101.85 | 102.84 | 0.00 | ORB-short ORB[102.45,103.95] vol=3.2x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:00:00 | 100.97 | 102.33 | 0.00 | T1 1.5R @ 100.97 |
| Target hit | 2024-05-28 15:20:00 | 100.00 | 101.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:35:00 | 103.93 | 104.54 | 0.00 | ORB-short ORB[104.00,105.48] vol=1.8x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 14:25:00 | 103.13 | 104.04 | 0.00 | T1 1.5R @ 103.13 |
| Target hit | 2024-06-10 15:20:00 | 101.68 | 103.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 102.99 | 103.37 | 0.00 | ORB-short ORB[103.05,104.07] vol=1.8x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:55:00 | 102.59 | 103.12 | 0.00 | T1 1.5R @ 102.59 |
| Stop hit — per-position SL triggered | 2024-06-13 14:45:00 | 102.99 | 102.67 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:10:00 | 101.95 | 102.60 | 0.00 | ORB-short ORB[102.25,103.30] vol=1.5x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-06-18 10:15:00 | 102.23 | 102.58 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:05:00 | 100.74 | 100.37 | 0.00 | ORB-long ORB[99.70,100.60] vol=3.0x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:25:00 | 101.27 | 100.48 | 0.00 | T1 1.5R @ 101.27 |
| Stop hit — per-position SL triggered | 2024-06-20 10:45:00 | 100.74 | 100.67 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:40:00 | 100.99 | 100.44 | 0.00 | ORB-long ORB[99.95,100.65] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2024-06-21 09:45:00 | 100.68 | 100.48 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 99.71 | 100.13 | 0.00 | ORB-short ORB[100.02,100.60] vol=2.0x ATR=0.16 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 99.87 | 100.12 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 11:15:00 | 99.37 | 99.69 | 0.00 | ORB-short ORB[99.55,100.35] vol=2.1x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:25:00 | 99.10 | 99.65 | 0.00 | T1 1.5R @ 99.10 |
| Stop hit — per-position SL triggered | 2024-06-26 11:40:00 | 99.37 | 99.61 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:40:00 | 101.45 | 100.09 | 0.00 | ORB-long ORB[98.69,99.47] vol=4.3x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 09:45:00 | 102.24 | 100.85 | 0.00 | T1 1.5R @ 102.24 |
| Stop hit — per-position SL triggered | 2024-06-28 09:50:00 | 101.45 | 100.95 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 99.95 | 99.69 | 0.00 | ORB-long ORB[99.36,99.80] vol=2.3x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-07-03 09:45:00 | 99.70 | 99.73 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:35:00 | 103.46 | 102.66 | 0.00 | ORB-long ORB[101.80,102.78] vol=3.8x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-07-04 10:20:00 | 103.02 | 103.15 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:35:00 | 106.54 | 105.41 | 0.00 | ORB-long ORB[103.70,104.99] vol=7.2x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:45:00 | 107.35 | 105.91 | 0.00 | T1 1.5R @ 107.35 |
| Stop hit — per-position SL triggered | 2024-07-09 09:50:00 | 106.54 | 105.98 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:45:00 | 103.50 | 102.10 | 0.00 | ORB-long ORB[101.02,102.35] vol=3.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-07-25 10:55:00 | 103.00 | 102.21 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:45:00 | 95.95 | 95.31 | 0.00 | ORB-long ORB[94.50,95.60] vol=1.5x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-08-19 09:55:00 | 95.61 | 95.37 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:10:00 | 97.71 | 98.03 | 0.00 | ORB-short ORB[97.75,98.60] vol=1.6x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:55:00 | 97.44 | 97.98 | 0.00 | T1 1.5R @ 97.44 |
| Stop hit — per-position SL triggered | 2024-08-23 13:25:00 | 97.71 | 97.92 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 94.79 | 95.14 | 0.00 | ORB-short ORB[94.94,95.50] vol=1.6x ATR=0.23 |
| Stop hit — per-position SL triggered | 2024-08-28 09:45:00 | 95.02 | 95.08 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:40:00 | 96.12 | 95.49 | 0.00 | ORB-long ORB[94.82,95.60] vol=2.8x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-08-29 09:45:00 | 95.87 | 96.07 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:15:00 | 97.77 | 98.37 | 0.00 | ORB-short ORB[98.45,98.98] vol=1.8x ATR=0.20 |
| Stop hit — per-position SL triggered | 2024-09-05 10:20:00 | 97.97 | 98.35 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 96.92 | 97.64 | 0.00 | ORB-short ORB[97.55,98.43] vol=2.1x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:55:00 | 96.55 | 97.43 | 0.00 | T1 1.5R @ 96.55 |
| Target hit | 2024-09-06 15:20:00 | 96.05 | 96.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-09-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:05:00 | 96.39 | 95.56 | 0.00 | ORB-long ORB[94.50,95.10] vol=3.4x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:15:00 | 96.83 | 95.84 | 0.00 | T1 1.5R @ 96.83 |
| Stop hit — per-position SL triggered | 2024-09-16 10:30:00 | 96.39 | 95.99 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 93.95 | 94.58 | 0.00 | ORB-short ORB[94.60,95.22] vol=1.7x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:50:00 | 93.56 | 94.44 | 0.00 | T1 1.5R @ 93.56 |
| Target hit | 2024-09-19 15:20:00 | 92.99 | 92.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:40:00 | 93.42 | 93.60 | 0.00 | ORB-short ORB[93.51,94.07] vol=1.6x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:55:00 | 93.09 | 93.54 | 0.00 | T1 1.5R @ 93.09 |
| Target hit | 2024-09-25 11:30:00 | 93.03 | 93.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-09-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:10:00 | 94.35 | 93.80 | 0.00 | ORB-long ORB[93.15,94.00] vol=1.7x ATR=0.27 |
| Stop hit — per-position SL triggered | 2024-09-27 11:40:00 | 94.08 | 94.17 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-10-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:45:00 | 94.00 | 93.59 | 0.00 | ORB-long ORB[92.90,93.99] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-10-03 10:05:00 | 93.71 | 93.71 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:00:00 | 90.71 | 92.01 | 0.00 | ORB-short ORB[92.41,93.25] vol=1.7x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 90.18 | 91.71 | 0.00 | T1 1.5R @ 90.18 |
| Target hit | 2024-10-07 11:30:00 | 90.06 | 89.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — BUY (started 2024-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:40:00 | 92.58 | 91.81 | 0.00 | ORB-long ORB[90.85,92.17] vol=2.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-10-09 10:00:00 | 92.24 | 91.97 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:00:00 | 88.99 | 89.66 | 0.00 | ORB-short ORB[89.74,90.25] vol=2.7x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:10:00 | 88.74 | 89.57 | 0.00 | T1 1.5R @ 88.74 |
| Target hit | 2024-10-16 15:20:00 | 87.97 | 88.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 85.39 | 84.67 | 0.00 | ORB-long ORB[83.91,85.15] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-11-07 09:45:00 | 84.98 | 84.77 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 82.65 | 82.31 | 0.00 | ORB-long ORB[81.70,82.48] vol=2.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2024-12-03 09:40:00 | 82.46 | 82.36 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 82.95 | 82.74 | 0.00 | ORB-long ORB[82.09,82.93] vol=3.5x ATR=0.15 |
| Stop hit — per-position SL triggered | 2024-12-04 09:40:00 | 82.80 | 82.76 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 83.32 | 82.80 | 0.00 | ORB-long ORB[81.97,83.00] vol=2.3x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:40:00 | 83.68 | 83.11 | 0.00 | T1 1.5R @ 83.68 |
| Target hit | 2024-12-06 12:05:00 | 85.32 | 85.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — BUY (started 2024-12-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:35:00 | 87.50 | 86.91 | 0.00 | ORB-long ORB[86.20,87.37] vol=3.2x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-12-09 09:40:00 | 87.14 | 86.94 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:25:00 | 85.96 | 86.38 | 0.00 | ORB-short ORB[86.22,86.65] vol=1.6x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:40:00 | 85.64 | 86.29 | 0.00 | T1 1.5R @ 85.64 |
| Target hit | 2024-12-12 15:20:00 | 85.15 | 85.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 81.50 | 81.91 | 0.00 | ORB-short ORB[81.74,82.53] vol=1.8x ATR=0.17 |
| Stop hit — per-position SL triggered | 2024-12-24 09:35:00 | 81.67 | 81.88 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:30:00 | 81.92 | 81.61 | 0.00 | ORB-long ORB[80.69,81.87] vol=2.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-01-01 09:50:00 | 81.60 | 81.71 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:05:00 | 81.42 | 81.76 | 0.00 | ORB-short ORB[81.62,82.49] vol=2.7x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-01-02 11:25:00 | 81.58 | 81.73 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:55:00 | 83.00 | 83.91 | 0.00 | ORB-short ORB[83.26,84.50] vol=1.7x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:30:00 | 82.52 | 83.73 | 0.00 | T1 1.5R @ 82.52 |
| Stop hit — per-position SL triggered | 2025-01-03 11:00:00 | 83.00 | 83.65 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 11:15:00 | 78.39 | 78.74 | 0.00 | ORB-short ORB[78.40,79.20] vol=2.0x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:45:00 | 78.14 | 78.56 | 0.00 | T1 1.5R @ 78.14 |
| Target hit | 2025-01-09 15:20:00 | 78.09 | 78.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 76.50 | 76.92 | 0.00 | ORB-short ORB[76.81,77.60] vol=1.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-01-15 09:35:00 | 76.81 | 76.90 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:35:00 | 79.04 | 78.54 | 0.00 | ORB-long ORB[77.94,78.74] vol=1.9x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-01-16 09:50:00 | 78.73 | 78.69 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:00:00 | 79.77 | 78.88 | 0.00 | ORB-long ORB[78.00,78.94] vol=1.5x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-01-17 10:05:00 | 79.54 | 78.96 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:45:00 | 79.75 | 80.00 | 0.00 | ORB-short ORB[79.80,80.19] vol=1.7x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:00:00 | 79.48 | 79.89 | 0.00 | T1 1.5R @ 79.48 |
| Target hit | 2025-01-21 11:35:00 | 79.49 | 79.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — BUY (started 2025-01-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:35:00 | 79.54 | 78.44 | 0.00 | ORB-long ORB[77.06,77.69] vol=1.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-01-23 10:50:00 | 79.25 | 78.56 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-02-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:50:00 | 79.74 | 79.45 | 0.00 | ORB-long ORB[78.54,79.51] vol=2.3x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-02-05 10:20:00 | 79.51 | 79.57 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 11:10:00 | 77.22 | 78.01 | 0.00 | ORB-short ORB[77.91,78.95] vol=1.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-02-24 11:40:00 | 77.52 | 77.91 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 78.21 | 77.76 | 0.00 | ORB-long ORB[77.24,78.19] vol=1.5x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:40:00 | 78.65 | 77.95 | 0.00 | T1 1.5R @ 78.65 |
| Stop hit — per-position SL triggered | 2025-02-25 10:05:00 | 78.21 | 78.10 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 11:15:00 | 77.85 | 78.34 | 0.00 | ORB-short ORB[78.03,79.00] vol=1.9x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 11:35:00 | 77.49 | 78.28 | 0.00 | T1 1.5R @ 77.49 |
| Stop hit — per-position SL triggered | 2025-03-13 12:05:00 | 77.85 | 78.21 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 81.19 | 80.64 | 0.00 | ORB-long ORB[80.02,80.93] vol=2.0x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:50:00 | 81.56 | 80.89 | 0.00 | T1 1.5R @ 81.56 |
| Target hit | 2025-03-21 15:00:00 | 82.88 | 83.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2025-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:35:00 | 85.83 | 85.41 | 0.00 | ORB-long ORB[84.84,85.60] vol=1.9x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 09:40:00 | 86.16 | 85.58 | 0.00 | T1 1.5R @ 86.16 |
| Stop hit — per-position SL triggered | 2025-04-16 09:45:00 | 85.83 | 85.62 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 86.20 | 85.42 | 0.00 | ORB-long ORB[84.85,85.50] vol=1.9x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:35:00 | 86.53 | 85.98 | 0.00 | T1 1.5R @ 86.53 |
| Target hit | 2025-04-21 15:20:00 | 87.73 | 86.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:45:00 | 83.35 | 83.73 | 0.00 | ORB-short ORB[83.36,84.50] vol=1.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-05-08 10:00:00 | 83.63 | 83.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-24 09:35:00 | 104.70 | 2024-05-24 09:40:00 | 105.18 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-05-28 09:35:00 | 101.85 | 2024-05-28 10:00:00 | 100.97 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2024-05-28 09:35:00 | 101.85 | 2024-05-28 15:20:00 | 100.00 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2024-06-10 09:35:00 | 103.93 | 2024-06-10 14:25:00 | 103.13 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-06-10 09:35:00 | 103.93 | 2024-06-10 15:20:00 | 101.68 | TARGET_HIT | 0.50 | 2.16% |
| SELL | retest1 | 2024-06-13 09:30:00 | 102.99 | 2024-06-13 09:55:00 | 102.59 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-13 09:30:00 | 102.99 | 2024-06-13 14:45:00 | 102.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 10:10:00 | 101.95 | 2024-06-18 10:15:00 | 102.23 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-20 10:05:00 | 100.74 | 2024-06-20 10:25:00 | 101.27 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-20 10:05:00 | 100.74 | 2024-06-20 10:45:00 | 100.74 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:40:00 | 100.99 | 2024-06-21 09:45:00 | 100.68 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-25 11:15:00 | 99.71 | 2024-06-25 11:20:00 | 99.87 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-06-26 11:15:00 | 99.37 | 2024-06-26 11:25:00 | 99.10 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-06-26 11:15:00 | 99.37 | 2024-06-26 11:40:00 | 99.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 09:40:00 | 101.45 | 2024-06-28 09:45:00 | 102.24 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-06-28 09:40:00 | 101.45 | 2024-06-28 09:50:00 | 101.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 09:30:00 | 99.95 | 2024-07-03 09:45:00 | 99.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-04 09:35:00 | 103.46 | 2024-07-04 10:20:00 | 103.02 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-09 09:35:00 | 106.54 | 2024-07-09 09:45:00 | 107.35 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-07-09 09:35:00 | 106.54 | 2024-07-09 09:50:00 | 106.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 10:45:00 | 103.50 | 2024-07-25 10:55:00 | 103.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-08-19 09:45:00 | 95.95 | 2024-08-19 09:55:00 | 95.61 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-23 11:10:00 | 97.71 | 2024-08-23 11:55:00 | 97.44 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-08-23 11:10:00 | 97.71 | 2024-08-23 13:25:00 | 97.71 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 94.79 | 2024-08-28 09:45:00 | 95.02 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-29 09:40:00 | 96.12 | 2024-08-29 09:45:00 | 95.87 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-05 10:15:00 | 97.77 | 2024-09-05 10:20:00 | 97.97 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-06 09:45:00 | 96.92 | 2024-09-06 09:55:00 | 96.55 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-06 09:45:00 | 96.92 | 2024-09-06 15:20:00 | 96.05 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2024-09-16 10:05:00 | 96.39 | 2024-09-16 10:15:00 | 96.83 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-09-16 10:05:00 | 96.39 | 2024-09-16 10:30:00 | 96.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:45:00 | 93.95 | 2024-09-19 09:50:00 | 93.56 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-09-19 09:45:00 | 93.95 | 2024-09-19 15:20:00 | 92.99 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-09-25 09:40:00 | 93.42 | 2024-09-25 09:55:00 | 93.09 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-09-25 09:40:00 | 93.42 | 2024-09-25 11:30:00 | 93.03 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-27 10:10:00 | 94.35 | 2024-09-27 11:40:00 | 94.08 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-03 09:45:00 | 94.00 | 2024-10-03 10:05:00 | 93.71 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-07 10:00:00 | 90.71 | 2024-10-07 10:15:00 | 90.18 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-10-07 10:00:00 | 90.71 | 2024-10-07 11:30:00 | 90.06 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2024-10-09 09:40:00 | 92.58 | 2024-10-09 10:00:00 | 92.24 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-16 11:00:00 | 88.99 | 2024-10-16 11:10:00 | 88.74 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-10-16 11:00:00 | 88.99 | 2024-10-16 15:20:00 | 87.97 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2024-11-07 09:35:00 | 85.39 | 2024-11-07 09:45:00 | 84.98 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-12-03 09:30:00 | 82.65 | 2024-12-03 09:40:00 | 82.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-04 09:30:00 | 82.95 | 2024-12-04 09:40:00 | 82.80 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-06 09:35:00 | 83.32 | 2024-12-06 09:40:00 | 83.68 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-06 09:35:00 | 83.32 | 2024-12-06 12:05:00 | 85.32 | TARGET_HIT | 0.50 | 2.40% |
| BUY | retest1 | 2024-12-09 09:35:00 | 87.50 | 2024-12-09 09:40:00 | 87.14 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-12 10:25:00 | 85.96 | 2024-12-12 10:40:00 | 85.64 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-12 10:25:00 | 85.96 | 2024-12-12 15:20:00 | 85.15 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-12-24 09:30:00 | 81.50 | 2024-12-24 09:35:00 | 81.67 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-01-01 09:30:00 | 81.92 | 2025-01-01 09:50:00 | 81.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-02 11:05:00 | 81.42 | 2025-01-02 11:25:00 | 81.58 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-01-03 09:55:00 | 83.00 | 2025-01-03 10:30:00 | 82.52 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-03 09:55:00 | 83.00 | 2025-01-03 11:00:00 | 83.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 11:15:00 | 78.39 | 2025-01-09 11:45:00 | 78.14 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-01-09 11:15:00 | 78.39 | 2025-01-09 15:20:00 | 78.09 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-15 09:30:00 | 76.50 | 2025-01-15 09:35:00 | 76.81 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-16 09:35:00 | 79.04 | 2025-01-16 09:50:00 | 78.73 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-17 10:00:00 | 79.77 | 2025-01-17 10:05:00 | 79.54 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-21 09:45:00 | 79.75 | 2025-01-21 10:00:00 | 79.48 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-01-21 09:45:00 | 79.75 | 2025-01-21 11:35:00 | 79.49 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2025-01-23 10:35:00 | 79.54 | 2025-01-23 10:50:00 | 79.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-02-05 09:50:00 | 79.74 | 2025-02-05 10:20:00 | 79.51 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-02-24 11:10:00 | 77.22 | 2025-02-24 11:40:00 | 77.52 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-25 09:30:00 | 78.21 | 2025-02-25 09:40:00 | 78.65 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-02-25 09:30:00 | 78.21 | 2025-02-25 10:05:00 | 78.21 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-13 11:15:00 | 77.85 | 2025-03-13 11:35:00 | 77.49 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-03-13 11:15:00 | 77.85 | 2025-03-13 12:05:00 | 77.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:35:00 | 81.19 | 2025-03-21 09:50:00 | 81.56 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-03-21 09:35:00 | 81.19 | 2025-03-21 15:00:00 | 82.88 | TARGET_HIT | 0.50 | 2.08% |
| BUY | retest1 | 2025-04-16 09:35:00 | 85.83 | 2025-04-16 09:40:00 | 86.16 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-04-16 09:35:00 | 85.83 | 2025-04-16 09:45:00 | 85.83 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:30:00 | 86.20 | 2025-04-21 09:35:00 | 86.53 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-04-21 09:30:00 | 86.20 | 2025-04-21 15:20:00 | 87.73 | TARGET_HIT | 0.50 | 1.77% |
| SELL | retest1 | 2025-05-08 09:45:00 | 83.35 | 2025-05-08 10:00:00 | 83.63 | STOP_HIT | 1.00 | -0.34% |
