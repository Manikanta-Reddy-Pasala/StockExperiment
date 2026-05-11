# JM Financial Ltd. (JMFINANCIL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 145.00
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
| ENTRY1 | 44 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 5 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 39
- **Target hits / Stop hits / Partials:** 5 / 39 / 19
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 10.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 16 | 43.2% | 3 | 21 | 13 | 0.28% | 10.5% |
| BUY @ 2nd Alert (retest1) | 37 | 16 | 43.2% | 3 | 21 | 13 | 0.28% | 10.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 26 | 8 | 30.8% | 2 | 18 | 6 | -0.01% | -0.4% |
| SELL @ 2nd Alert (retest1) | 26 | 8 | 30.8% | 2 | 18 | 6 | -0.01% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 63 | 24 | 38.1% | 5 | 39 | 19 | 0.16% | 10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:00:00 | 79.45 | 80.08 | 0.00 | ORB-short ORB[79.85,80.70] vol=2.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 79.78 | 80.02 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:50:00 | 81.80 | 81.20 | 0.00 | ORB-long ORB[80.40,81.40] vol=2.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2024-05-15 09:55:00 | 81.47 | 81.24 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:30:00 | 81.65 | 81.92 | 0.00 | ORB-short ORB[81.80,82.60] vol=3.0x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-05-17 09:45:00 | 81.94 | 81.85 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 81.05 | 81.86 | 0.00 | ORB-short ORB[81.80,82.40] vol=1.8x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 81.43 | 81.77 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:05:00 | 80.75 | 80.85 | 0.00 | ORB-short ORB[80.80,81.55] vol=1.7x ATR=0.26 |
| Stop hit — per-position SL triggered | 2024-05-30 11:15:00 | 81.01 | 80.88 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:50:00 | 79.55 | 80.16 | 0.00 | ORB-short ORB[80.05,80.80] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2024-05-31 09:55:00 | 79.88 | 80.07 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:55:00 | 79.45 | 79.87 | 0.00 | ORB-short ORB[79.65,80.50] vol=2.2x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-06-10 10:10:00 | 79.75 | 79.76 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 10:00:00 | 78.00 | 78.28 | 0.00 | ORB-short ORB[78.20,79.07] vol=1.5x ATR=0.21 |
| Stop hit — per-position SL triggered | 2024-06-11 10:35:00 | 78.21 | 78.22 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 86.14 | 85.15 | 0.00 | ORB-long ORB[84.28,85.25] vol=2.3x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-06-14 10:00:00 | 85.76 | 85.21 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:00:00 | 84.74 | 84.15 | 0.00 | ORB-long ORB[83.66,84.50] vol=1.8x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-06-18 10:05:00 | 84.37 | 84.18 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:35:00 | 84.02 | 83.44 | 0.00 | ORB-long ORB[82.70,83.50] vol=8.7x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-06-25 10:40:00 | 83.63 | 83.46 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:00:00 | 85.16 | 84.57 | 0.00 | ORB-long ORB[84.16,84.86] vol=2.4x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-06-27 10:05:00 | 84.79 | 84.59 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:40:00 | 87.88 | 86.58 | 0.00 | ORB-long ORB[85.08,85.99] vol=4.7x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-06-28 09:45:00 | 87.51 | 86.70 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:30:00 | 89.38 | 88.21 | 0.00 | ORB-long ORB[87.70,88.70] vol=2.2x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-07-01 10:35:00 | 88.96 | 88.28 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 91.46 | 91.15 | 0.00 | ORB-long ORB[90.43,91.45] vol=1.9x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:30:00 | 91.99 | 91.44 | 0.00 | T1 1.5R @ 91.99 |
| Stop hit — per-position SL triggered | 2024-07-04 10:35:00 | 91.46 | 91.44 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:30:00 | 94.14 | 94.64 | 0.00 | ORB-short ORB[94.20,95.50] vol=1.9x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-07-10 09:35:00 | 94.57 | 94.72 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:45:00 | 94.02 | 93.51 | 0.00 | ORB-long ORB[92.96,93.67] vol=4.2x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 93.60 | 93.60 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:30:00 | 93.82 | 94.37 | 0.00 | ORB-short ORB[94.36,95.20] vol=3.3x ATR=0.31 |
| Stop hit — per-position SL triggered | 2024-07-23 09:35:00 | 94.13 | 94.36 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:30:00 | 103.45 | 102.50 | 0.00 | ORB-long ORB[101.76,102.93] vol=2.0x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:35:00 | 104.20 | 102.86 | 0.00 | T1 1.5R @ 104.20 |
| Stop hit — per-position SL triggered | 2024-07-30 09:40:00 | 103.45 | 102.92 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 105.18 | 104.88 | 0.00 | ORB-long ORB[103.76,104.85] vol=6.0x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 104.67 | 104.93 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:05:00 | 105.35 | 104.74 | 0.00 | ORB-long ORB[104.27,105.00] vol=2.7x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-08-01 10:25:00 | 104.96 | 104.85 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:10:00 | 99.31 | 100.56 | 0.00 | ORB-short ORB[100.50,101.82] vol=3.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-08-06 11:55:00 | 99.81 | 100.45 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:10:00 | 90.26 | 90.87 | 0.00 | ORB-short ORB[91.49,92.34] vol=5.8x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-08-16 10:20:00 | 90.68 | 90.85 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:40:00 | 93.80 | 94.98 | 0.00 | ORB-short ORB[94.10,95.50] vol=1.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:50:00 | 93.20 | 94.86 | 0.00 | T1 1.5R @ 93.20 |
| Target hit | 2024-08-21 15:20:00 | 92.67 | 93.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2024-08-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:05:00 | 91.95 | 92.34 | 0.00 | ORB-short ORB[92.56,93.18] vol=2.2x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-08-23 10:40:00 | 92.20 | 92.26 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:40:00 | 93.86 | 92.83 | 0.00 | ORB-long ORB[92.31,93.24] vol=2.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:45:00 | 94.44 | 93.48 | 0.00 | T1 1.5R @ 94.44 |
| Stop hit — per-position SL triggered | 2024-08-26 10:00:00 | 93.86 | 93.56 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:35:00 | 103.03 | 102.35 | 0.00 | ORB-long ORB[101.29,102.71] vol=1.9x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:40:00 | 103.78 | 105.79 | 0.00 | T1 1.5R @ 103.78 |
| Target hit | 2024-08-29 10:35:00 | 109.02 | 109.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 129.70 | 128.53 | 0.00 | ORB-long ORB[127.16,128.78] vol=3.4x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 09:35:00 | 130.53 | 128.81 | 0.00 | T1 1.5R @ 130.53 |
| Stop hit — per-position SL triggered | 2024-09-24 09:40:00 | 129.70 | 128.90 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:05:00 | 148.08 | 145.40 | 0.00 | ORB-long ORB[143.80,145.85] vol=4.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-10-11 10:20:00 | 147.02 | 145.84 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:30:00 | 130.20 | 131.05 | 0.00 | ORB-short ORB[130.57,132.50] vol=2.0x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-11-21 09:40:00 | 130.86 | 130.83 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:35:00 | 140.52 | 139.60 | 0.00 | ORB-long ORB[138.10,139.85] vol=2.1x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:05:00 | 141.68 | 140.12 | 0.00 | T1 1.5R @ 141.68 |
| Target hit | 2024-12-03 11:35:00 | 140.80 | 140.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-12-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:20:00 | 140.78 | 139.84 | 0.00 | ORB-long ORB[139.05,140.45] vol=1.7x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:55:00 | 141.89 | 140.42 | 0.00 | T1 1.5R @ 141.89 |
| Stop hit — per-position SL triggered | 2024-12-06 11:20:00 | 140.78 | 140.57 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:50:00 | 141.99 | 140.78 | 0.00 | ORB-long ORB[140.15,141.68] vol=2.7x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:55:00 | 142.81 | 141.77 | 0.00 | T1 1.5R @ 142.81 |
| Stop hit — per-position SL triggered | 2024-12-12 10:10:00 | 141.99 | 141.84 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 136.94 | 138.60 | 0.00 | ORB-short ORB[138.26,139.90] vol=1.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:25:00 | 136.00 | 138.18 | 0.00 | T1 1.5R @ 136.00 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 136.94 | 137.70 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:15:00 | 135.15 | 134.34 | 0.00 | ORB-long ORB[133.00,135.00] vol=2.2x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:35:00 | 136.07 | 134.65 | 0.00 | T1 1.5R @ 136.07 |
| Stop hit — per-position SL triggered | 2024-12-20 10:45:00 | 135.15 | 134.72 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 123.76 | 124.74 | 0.00 | ORB-short ORB[124.33,126.04] vol=2.3x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:05:00 | 122.96 | 124.18 | 0.00 | T1 1.5R @ 122.96 |
| Stop hit — per-position SL triggered | 2024-12-26 12:35:00 | 123.76 | 123.53 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:40:00 | 126.85 | 126.08 | 0.00 | ORB-long ORB[125.20,126.55] vol=2.1x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 12:00:00 | 127.74 | 126.80 | 0.00 | T1 1.5R @ 127.74 |
| Target hit | 2024-12-27 15:20:00 | 127.68 | 127.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 11:10:00 | 128.50 | 129.60 | 0.00 | ORB-short ORB[128.96,130.55] vol=1.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 12:10:00 | 127.82 | 129.35 | 0.00 | T1 1.5R @ 127.82 |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 128.50 | 129.04 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:30:00 | 122.86 | 122.08 | 0.00 | ORB-long ORB[121.23,122.45] vol=3.1x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:10:00 | 123.67 | 122.60 | 0.00 | T1 1.5R @ 123.67 |
| Stop hit — per-position SL triggered | 2025-01-09 10:40:00 | 122.86 | 122.72 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-01-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:40:00 | 117.94 | 118.52 | 0.00 | ORB-short ORB[117.96,119.60] vol=2.0x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:50:00 | 117.35 | 118.43 | 0.00 | T1 1.5R @ 117.35 |
| Target hit | 2025-01-17 12:15:00 | 117.49 | 117.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2025-01-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:30:00 | 117.68 | 119.67 | 0.00 | ORB-short ORB[119.73,121.48] vol=1.8x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-01-21 10:35:00 | 118.22 | 119.62 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 115.17 | 113.91 | 0.00 | ORB-long ORB[112.75,113.95] vol=1.8x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:35:00 | 116.16 | 114.59 | 0.00 | T1 1.5R @ 116.16 |
| Stop hit — per-position SL triggered | 2025-01-23 11:40:00 | 115.17 | 114.73 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 112.99 | 114.25 | 0.00 | ORB-short ORB[113.93,115.49] vol=2.2x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:45:00 | 112.26 | 113.85 | 0.00 | T1 1.5R @ 112.26 |
| Stop hit — per-position SL triggered | 2025-01-24 10:05:00 | 112.99 | 113.16 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:45:00 | 101.58 | 100.96 | 0.00 | ORB-long ORB[100.23,101.50] vol=1.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:50:00 | 102.12 | 101.14 | 0.00 | T1 1.5R @ 102.12 |
| Stop hit — per-position SL triggered | 2025-04-21 09:55:00 | 101.58 | 101.18 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:00:00 | 79.45 | 2024-05-14 10:15:00 | 79.78 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-05-15 09:50:00 | 81.80 | 2024-05-15 09:55:00 | 81.47 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-17 09:30:00 | 81.65 | 2024-05-17 09:45:00 | 81.94 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-22 09:40:00 | 81.05 | 2024-05-22 09:50:00 | 81.43 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-05-30 11:05:00 | 80.75 | 2024-05-30 11:15:00 | 81.01 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-31 09:50:00 | 79.55 | 2024-05-31 09:55:00 | 79.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-06-10 09:55:00 | 79.45 | 2024-06-10 10:10:00 | 79.75 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-11 10:00:00 | 78.00 | 2024-06-11 10:35:00 | 78.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-14 09:55:00 | 86.14 | 2024-06-14 10:00:00 | 85.76 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-06-18 10:00:00 | 84.74 | 2024-06-18 10:05:00 | 84.37 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-06-25 10:35:00 | 84.02 | 2024-06-25 10:40:00 | 83.63 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-06-27 10:00:00 | 85.16 | 2024-06-27 10:05:00 | 84.79 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-06-28 09:40:00 | 87.88 | 2024-06-28 09:45:00 | 87.51 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-01 10:30:00 | 89.38 | 2024-07-01 10:35:00 | 88.96 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-07-04 09:30:00 | 91.46 | 2024-07-04 10:30:00 | 91.99 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-04 09:30:00 | 91.46 | 2024-07-04 10:35:00 | 91.46 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 09:30:00 | 94.14 | 2024-07-10 09:35:00 | 94.57 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-07-12 09:45:00 | 94.02 | 2024-07-12 09:55:00 | 93.60 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-07-23 09:30:00 | 93.82 | 2024-07-23 09:35:00 | 94.13 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-30 09:30:00 | 103.45 | 2024-07-30 09:35:00 | 104.20 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-07-30 09:30:00 | 103.45 | 2024-07-30 09:40:00 | 103.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 09:30:00 | 105.18 | 2024-07-31 09:45:00 | 104.67 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-08-01 10:05:00 | 105.35 | 2024-08-01 10:25:00 | 104.96 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-06 11:10:00 | 99.31 | 2024-08-06 11:55:00 | 99.81 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-08-16 10:10:00 | 90.26 | 2024-08-16 10:20:00 | 90.68 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-08-21 10:40:00 | 93.80 | 2024-08-21 10:50:00 | 93.20 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-08-21 10:40:00 | 93.80 | 2024-08-21 15:20:00 | 92.67 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2024-08-23 10:05:00 | 91.95 | 2024-08-23 10:40:00 | 92.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-26 09:40:00 | 93.86 | 2024-08-26 09:45:00 | 94.44 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-08-26 09:40:00 | 93.86 | 2024-08-26 10:00:00 | 93.86 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:35:00 | 103.03 | 2024-08-29 09:40:00 | 103.78 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-08-29 09:35:00 | 103.03 | 2024-08-29 10:35:00 | 109.02 | TARGET_HIT | 0.50 | 5.81% |
| BUY | retest1 | 2024-09-24 09:30:00 | 129.70 | 2024-09-24 09:35:00 | 130.53 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-09-24 09:30:00 | 129.70 | 2024-09-24 09:40:00 | 129.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 10:05:00 | 148.08 | 2024-10-11 10:20:00 | 147.02 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2024-11-21 09:30:00 | 130.20 | 2024-11-21 09:40:00 | 130.86 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-12-03 09:35:00 | 140.52 | 2024-12-03 10:05:00 | 141.68 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-12-03 09:35:00 | 140.52 | 2024-12-03 11:35:00 | 140.80 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-12-06 10:20:00 | 140.78 | 2024-12-06 10:55:00 | 141.89 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-12-06 10:20:00 | 140.78 | 2024-12-06 11:20:00 | 140.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-12 09:50:00 | 141.99 | 2024-12-12 09:55:00 | 142.81 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-12-12 09:50:00 | 141.99 | 2024-12-12 10:10:00 | 141.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:15:00 | 136.94 | 2024-12-13 10:25:00 | 136.00 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-12-13 10:15:00 | 136.94 | 2024-12-13 10:50:00 | 136.94 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 10:15:00 | 135.15 | 2024-12-20 10:35:00 | 136.07 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-20 10:15:00 | 135.15 | 2024-12-20 10:45:00 | 135.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 09:30:00 | 123.76 | 2024-12-26 10:05:00 | 122.96 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-12-26 09:30:00 | 123.76 | 2024-12-26 12:35:00 | 123.76 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 09:40:00 | 126.85 | 2024-12-27 12:00:00 | 127.74 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-12-27 09:40:00 | 126.85 | 2024-12-27 15:20:00 | 127.68 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2025-01-01 11:10:00 | 128.50 | 2025-01-01 12:10:00 | 127.82 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-01 11:10:00 | 128.50 | 2025-01-01 13:15:00 | 128.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 09:30:00 | 122.86 | 2025-01-09 10:10:00 | 123.67 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-09 09:30:00 | 122.86 | 2025-01-09 10:40:00 | 122.86 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-17 10:40:00 | 117.94 | 2025-01-17 10:50:00 | 117.35 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-01-17 10:40:00 | 117.94 | 2025-01-17 12:15:00 | 117.49 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-21 10:30:00 | 117.68 | 2025-01-21 10:35:00 | 118.22 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-23 09:35:00 | 115.17 | 2025-01-23 10:35:00 | 116.16 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2025-01-23 09:35:00 | 115.17 | 2025-01-23 11:40:00 | 115.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:35:00 | 112.99 | 2025-01-24 09:45:00 | 112.26 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-01-24 09:35:00 | 112.99 | 2025-01-24 10:05:00 | 112.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:45:00 | 101.58 | 2025-04-21 09:50:00 | 102.12 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-21 09:45:00 | 101.58 | 2025-04-21 09:55:00 | 101.58 | STOP_HIT | 0.50 | 0.00% |
