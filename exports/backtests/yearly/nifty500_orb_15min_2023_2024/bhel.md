# Bharat Heavy Electricals Ltd. (BHEL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53781 bars)
- **Last close:** 403.20
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 14 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 66
- **Target hits / Stop hits / Partials:** 14 / 65 / 30
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 12.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 78 | 35 | 44.9% | 12 | 42 | 24 | 0.17% | 13.0% |
| BUY @ 2nd Alert (retest1) | 78 | 35 | 44.9% | 12 | 42 | 24 | 0.17% | 13.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 31 | 8 | 25.8% | 2 | 23 | 6 | -0.03% | -0.9% |
| SELL @ 2nd Alert (retest1) | 31 | 8 | 25.8% | 2 | 23 | 6 | -0.03% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 109 | 43 | 39.4% | 14 | 65 | 30 | 0.11% | 12.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 11:00:00 | 81.45 | 80.79 | 0.00 | ORB-long ORB[80.40,81.30] vol=3.1x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-05-12 15:00:00 | 81.09 | 81.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-15 11:15:00 | 80.40 | 80.72 | 0.00 | ORB-short ORB[80.50,81.30] vol=1.9x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-05-15 12:05:00 | 80.59 | 80.67 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:45:00 | 81.70 | 81.29 | 0.00 | ORB-long ORB[80.55,81.50] vol=1.9x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 09:50:00 | 82.02 | 81.51 | 0.00 | T1 1.5R @ 82.02 |
| Stop hit — per-position SL triggered | 2023-05-16 10:15:00 | 81.70 | 81.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 80.35 | 81.13 | 0.00 | ORB-short ORB[80.85,81.85] vol=3.6x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 09:35:00 | 79.84 | 80.99 | 0.00 | T1 1.5R @ 79.84 |
| Stop hit — per-position SL triggered | 2023-05-19 10:15:00 | 80.35 | 80.75 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:30:00 | 80.55 | 80.30 | 0.00 | ORB-long ORB[79.90,80.50] vol=1.8x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-24 09:35:00 | 80.86 | 80.42 | 0.00 | T1 1.5R @ 80.86 |
| Stop hit — per-position SL triggered | 2023-05-24 10:20:00 | 80.55 | 80.53 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 11:00:00 | 80.70 | 80.38 | 0.00 | ORB-long ORB[80.00,80.50] vol=2.7x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 11:05:00 | 81.01 | 80.44 | 0.00 | T1 1.5R @ 81.01 |
| Stop hit — per-position SL triggered | 2023-05-25 11:15:00 | 80.70 | 80.47 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 09:30:00 | 83.90 | 83.24 | 0.00 | ORB-long ORB[82.30,83.50] vol=1.7x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-05-30 09:35:00 | 83.51 | 83.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 10:00:00 | 82.85 | 82.45 | 0.00 | ORB-long ORB[82.00,82.55] vol=3.3x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 10:20:00 | 83.22 | 82.72 | 0.00 | T1 1.5R @ 83.22 |
| Target hit | 2023-06-02 12:05:00 | 83.05 | 83.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2023-06-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:50:00 | 84.70 | 84.06 | 0.00 | ORB-long ORB[83.35,84.35] vol=3.4x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 11:45:00 | 85.18 | 84.53 | 0.00 | T1 1.5R @ 85.18 |
| Stop hit — per-position SL triggered | 2023-06-05 13:05:00 | 84.70 | 84.66 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 11:05:00 | 85.40 | 84.73 | 0.00 | ORB-long ORB[84.10,84.85] vol=7.5x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 11:15:00 | 85.75 | 84.98 | 0.00 | T1 1.5R @ 85.75 |
| Stop hit — per-position SL triggered | 2023-06-08 11:20:00 | 85.40 | 85.01 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 83.60 | 84.37 | 0.00 | ORB-short ORB[84.00,84.95] vol=2.3x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 09:35:00 | 83.13 | 84.22 | 0.00 | T1 1.5R @ 83.13 |
| Stop hit — per-position SL triggered | 2023-06-09 09:45:00 | 83.60 | 84.00 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 09:45:00 | 86.55 | 85.99 | 0.00 | ORB-long ORB[85.20,86.35] vol=1.6x ATR=0.40 |
| Stop hit — per-position SL triggered | 2023-06-12 09:55:00 | 86.15 | 86.06 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 11:15:00 | 84.50 | 84.08 | 0.00 | ORB-long ORB[83.90,84.30] vol=4.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-06-15 11:20:00 | 84.31 | 84.09 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:10:00 | 87.10 | 87.67 | 0.00 | ORB-short ORB[87.30,88.40] vol=2.0x ATR=0.22 |
| Stop hit — per-position SL triggered | 2023-06-21 11:20:00 | 87.32 | 87.61 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 09:55:00 | 85.75 | 86.55 | 0.00 | ORB-short ORB[86.50,87.20] vol=1.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2023-06-22 10:10:00 | 86.10 | 86.34 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:30:00 | 86.40 | 85.93 | 0.00 | ORB-long ORB[85.10,86.35] vol=1.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-06-30 09:40:00 | 86.11 | 85.97 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 10:50:00 | 88.85 | 87.93 | 0.00 | ORB-long ORB[87.40,88.55] vol=3.1x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-07-03 10:55:00 | 88.53 | 88.01 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:55:00 | 86.90 | 87.84 | 0.00 | ORB-short ORB[87.75,88.75] vol=2.0x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-07-04 10:05:00 | 87.26 | 87.65 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:35:00 | 88.75 | 88.35 | 0.00 | ORB-long ORB[87.10,88.30] vol=7.0x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 09:40:00 | 89.23 | 88.99 | 0.00 | T1 1.5R @ 89.23 |
| Target hit | 2023-07-05 15:20:00 | 93.15 | 91.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2023-07-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:30:00 | 91.15 | 92.19 | 0.00 | ORB-short ORB[91.80,92.75] vol=1.5x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 10:35:00 | 90.70 | 92.03 | 0.00 | T1 1.5R @ 90.70 |
| Target hit | 2023-07-07 13:30:00 | 90.90 | 90.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — BUY (started 2023-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:40:00 | 92.85 | 92.12 | 0.00 | ORB-long ORB[91.50,92.25] vol=3.1x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 11:00:00 | 93.40 | 92.69 | 0.00 | T1 1.5R @ 93.40 |
| Target hit | 2023-07-11 15:20:00 | 93.95 | 93.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2023-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:50:00 | 95.70 | 94.95 | 0.00 | ORB-long ORB[94.30,95.15] vol=4.0x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-07-12 09:55:00 | 95.41 | 95.04 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 10:30:00 | 93.15 | 92.76 | 0.00 | ORB-long ORB[92.20,93.05] vol=1.8x ATR=0.27 |
| Stop hit — per-position SL triggered | 2023-07-17 11:15:00 | 92.88 | 92.86 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:50:00 | 93.65 | 92.97 | 0.00 | ORB-long ORB[92.35,93.40] vol=2.1x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 09:55:00 | 94.16 | 93.15 | 0.00 | T1 1.5R @ 94.16 |
| Target hit | 2023-07-18 11:35:00 | 94.55 | 94.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 11:15:00 | 94.40 | 95.13 | 0.00 | ORB-short ORB[94.70,95.55] vol=2.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-07-19 11:25:00 | 94.65 | 95.12 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:30:00 | 95.95 | 95.60 | 0.00 | ORB-long ORB[95.00,95.80] vol=2.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 09:35:00 | 96.33 | 95.87 | 0.00 | T1 1.5R @ 96.33 |
| Target hit | 2023-07-21 09:35:00 | 95.85 | 95.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2023-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:50:00 | 96.20 | 95.74 | 0.00 | ORB-long ORB[94.80,95.80] vol=1.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-07-24 11:25:00 | 95.90 | 95.81 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-07-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:45:00 | 99.15 | 98.75 | 0.00 | ORB-long ORB[97.80,99.00] vol=2.7x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-07-26 10:10:00 | 98.78 | 98.88 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-07-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 09:50:00 | 102.70 | 102.04 | 0.00 | ORB-long ORB[101.55,102.30] vol=1.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 10:05:00 | 103.25 | 102.28 | 0.00 | T1 1.5R @ 103.25 |
| Stop hit — per-position SL triggered | 2023-07-27 11:00:00 | 102.70 | 102.70 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:50:00 | 103.05 | 103.96 | 0.00 | ORB-short ORB[103.25,104.60] vol=1.9x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:00:00 | 102.54 | 103.76 | 0.00 | T1 1.5R @ 102.54 |
| Target hit | 2023-08-02 15:20:00 | 100.55 | 101.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2023-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 09:45:00 | 98.65 | 97.94 | 0.00 | ORB-long ORB[97.25,98.30] vol=2.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 09:50:00 | 99.24 | 98.92 | 0.00 | T1 1.5R @ 99.24 |
| Target hit | 2023-08-08 10:45:00 | 99.15 | 99.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2023-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:45:00 | 101.00 | 100.54 | 0.00 | ORB-long ORB[99.85,100.80] vol=1.6x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 10:25:00 | 101.47 | 101.30 | 0.00 | T1 1.5R @ 101.47 |
| Target hit | 2023-08-11 11:10:00 | 101.50 | 101.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2023-08-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-16 10:00:00 | 100.25 | 100.87 | 0.00 | ORB-short ORB[100.50,101.45] vol=2.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-08-16 10:10:00 | 100.64 | 100.80 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:30:00 | 100.55 | 100.23 | 0.00 | ORB-long ORB[99.60,100.35] vol=2.3x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 09:40:00 | 101.06 | 100.60 | 0.00 | T1 1.5R @ 101.06 |
| Stop hit — per-position SL triggered | 2023-08-17 10:55:00 | 100.55 | 100.84 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-08-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:35:00 | 99.35 | 99.91 | 0.00 | ORB-short ORB[99.65,100.35] vol=1.7x ATR=0.27 |
| Stop hit — per-position SL triggered | 2023-08-18 09:45:00 | 99.62 | 99.82 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-08-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 10:05:00 | 103.20 | 102.26 | 0.00 | ORB-long ORB[101.05,102.35] vol=3.3x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-08-22 10:10:00 | 102.81 | 102.30 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:35:00 | 142.30 | 140.72 | 0.00 | ORB-long ORB[138.25,139.75] vol=5.0x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-09-05 09:40:00 | 141.36 | 140.90 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:30:00 | 136.75 | 136.18 | 0.00 | ORB-long ORB[135.00,136.55] vol=2.0x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 09:35:00 | 137.43 | 136.37 | 0.00 | T1 1.5R @ 137.43 |
| Target hit | 2023-09-07 11:30:00 | 137.05 | 137.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 11:15:00 | 130.05 | 130.74 | 0.00 | ORB-short ORB[130.10,132.05] vol=2.9x ATR=0.38 |
| Stop hit — per-position SL triggered | 2023-09-15 11:25:00 | 130.43 | 130.71 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 09:30:00 | 123.90 | 123.28 | 0.00 | ORB-long ORB[122.25,123.75] vol=2.3x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-09-21 09:35:00 | 123.42 | 123.32 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:50:00 | 128.30 | 127.41 | 0.00 | ORB-long ORB[126.70,127.65] vol=3.4x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 09:55:00 | 129.22 | 127.61 | 0.00 | T1 1.5R @ 129.22 |
| Stop hit — per-position SL triggered | 2023-09-28 10:00:00 | 128.30 | 127.61 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 10:30:00 | 132.10 | 130.95 | 0.00 | ORB-long ORB[130.10,131.85] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-10-03 10:50:00 | 131.49 | 131.05 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 10:20:00 | 129.30 | 128.56 | 0.00 | ORB-long ORB[127.85,128.70] vol=2.6x ATR=0.40 |
| Stop hit — per-position SL triggered | 2023-10-06 10:25:00 | 128.90 | 128.59 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 09:55:00 | 128.45 | 129.15 | 0.00 | ORB-short ORB[129.00,130.20] vol=3.1x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 10:00:00 | 127.78 | 128.91 | 0.00 | T1 1.5R @ 127.78 |
| Stop hit — per-position SL triggered | 2023-10-13 10:25:00 | 128.45 | 128.80 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 11:00:00 | 129.00 | 128.00 | 0.00 | ORB-long ORB[127.00,128.75] vol=2.4x ATR=0.38 |
| Stop hit — per-position SL triggered | 2023-10-16 11:15:00 | 128.62 | 128.15 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 10:00:00 | 132.55 | 131.93 | 0.00 | ORB-long ORB[131.15,132.40] vol=1.7x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 10:15:00 | 133.24 | 132.35 | 0.00 | T1 1.5R @ 133.24 |
| Stop hit — per-position SL triggered | 2023-10-17 10:20:00 | 132.55 | 132.38 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:45:00 | 133.40 | 132.81 | 0.00 | ORB-long ORB[132.00,133.00] vol=3.4x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-10-18 10:45:00 | 132.91 | 133.08 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:40:00 | 113.90 | 114.51 | 0.00 | ORB-short ORB[114.15,115.40] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-10-26 09:50:00 | 114.52 | 114.45 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 10:05:00 | 120.25 | 119.17 | 0.00 | ORB-long ORB[118.10,119.40] vol=1.7x ATR=0.51 |
| Stop hit — per-position SL triggered | 2023-10-27 10:10:00 | 119.74 | 119.22 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 09:40:00 | 121.60 | 121.26 | 0.00 | ORB-long ORB[120.70,121.45] vol=1.5x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-11-01 09:45:00 | 121.16 | 121.27 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:25:00 | 124.40 | 123.42 | 0.00 | ORB-long ORB[122.35,123.60] vol=2.3x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 12:00:00 | 125.14 | 123.99 | 0.00 | T1 1.5R @ 125.14 |
| Stop hit — per-position SL triggered | 2023-11-02 12:10:00 | 124.40 | 124.02 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 09:30:00 | 130.35 | 129.74 | 0.00 | ORB-long ORB[128.65,130.25] vol=1.8x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 09:40:00 | 131.31 | 130.18 | 0.00 | T1 1.5R @ 131.31 |
| Target hit | 2023-11-03 10:10:00 | 130.80 | 130.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:15:00 | 130.50 | 129.22 | 0.00 | ORB-long ORB[128.70,129.90] vol=3.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-11-06 10:50:00 | 129.93 | 129.59 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 09:45:00 | 128.95 | 129.60 | 0.00 | ORB-short ORB[129.30,130.25] vol=1.8x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-11-08 09:55:00 | 129.32 | 129.57 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:45:00 | 140.30 | 141.21 | 0.00 | ORB-short ORB[141.30,142.60] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-11-20 10:25:00 | 140.91 | 141.05 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-12-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:40:00 | 178.55 | 177.02 | 0.00 | ORB-long ORB[175.80,177.70] vol=2.4x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 12:05:00 | 179.85 | 177.56 | 0.00 | T1 1.5R @ 179.85 |
| Stop hit — per-position SL triggered | 2023-12-06 12:20:00 | 178.55 | 177.92 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:40:00 | 180.40 | 178.83 | 0.00 | ORB-long ORB[177.05,179.20] vol=2.2x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-12-11 10:00:00 | 179.40 | 179.38 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-12-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:10:00 | 177.25 | 178.98 | 0.00 | ORB-short ORB[179.50,180.75] vol=1.5x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-12-12 10:30:00 | 177.88 | 178.67 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 10:55:00 | 181.35 | 178.25 | 0.00 | ORB-long ORB[176.55,178.70] vol=4.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-12-13 11:15:00 | 180.48 | 179.06 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:35:00 | 188.95 | 187.33 | 0.00 | ORB-long ORB[185.65,188.00] vol=3.2x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 09:40:00 | 190.26 | 188.34 | 0.00 | T1 1.5R @ 190.26 |
| Stop hit — per-position SL triggered | 2023-12-20 09:45:00 | 188.95 | 188.47 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:45:00 | 180.90 | 179.99 | 0.00 | ORB-long ORB[178.65,180.85] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-12-26 10:35:00 | 179.99 | 180.50 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:40:00 | 184.45 | 183.22 | 0.00 | ORB-long ORB[182.00,183.50] vol=2.8x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:45:00 | 185.63 | 184.98 | 0.00 | T1 1.5R @ 185.63 |
| Target hit | 2023-12-28 10:00:00 | 187.30 | 187.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2024-01-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:50:00 | 197.10 | 196.53 | 0.00 | ORB-long ORB[195.05,196.60] vol=3.6x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-01-05 10:00:00 | 196.50 | 196.56 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-01-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 09:55:00 | 199.20 | 196.19 | 0.00 | ORB-long ORB[194.70,196.60] vol=3.9x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-01-08 10:00:00 | 198.08 | 196.46 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 09:35:00 | 220.20 | 223.72 | 0.00 | ORB-short ORB[223.80,225.50] vol=1.8x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-01-20 09:40:00 | 221.43 | 223.45 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-01-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 09:55:00 | 215.30 | 212.95 | 0.00 | ORB-long ORB[211.00,213.80] vol=2.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-01-25 10:00:00 | 214.21 | 213.04 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-01-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-31 11:10:00 | 226.90 | 229.60 | 0.00 | ORB-short ORB[228.00,230.80] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 11:25:00 | 225.49 | 229.14 | 0.00 | T1 1.5R @ 225.49 |
| Stop hit — per-position SL triggered | 2024-01-31 12:25:00 | 226.90 | 228.68 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:30:00 | 232.05 | 230.73 | 0.00 | ORB-long ORB[229.00,231.40] vol=1.9x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 09:35:00 | 233.74 | 231.67 | 0.00 | T1 1.5R @ 233.74 |
| Target hit | 2024-02-02 10:50:00 | 232.55 | 232.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 69 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 230.55 | 232.80 | 0.00 | ORB-short ORB[231.85,234.35] vol=2.0x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-02-08 11:25:00 | 231.49 | 232.40 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 10:50:00 | 229.50 | 227.96 | 0.00 | ORB-long ORB[226.40,229.00] vol=2.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-02-16 11:25:00 | 228.67 | 228.39 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-02-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:40:00 | 225.75 | 229.54 | 0.00 | ORB-short ORB[229.55,232.35] vol=2.0x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-02-21 12:50:00 | 226.68 | 227.90 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 11:05:00 | 224.65 | 226.48 | 0.00 | ORB-short ORB[226.50,228.40] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-02-27 11:15:00 | 225.39 | 226.36 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-03-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:55:00 | 259.45 | 257.72 | 0.00 | ORB-long ORB[254.80,258.40] vol=1.7x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-03-07 11:25:00 | 258.14 | 257.81 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-04-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:35:00 | 248.60 | 250.43 | 0.00 | ORB-short ORB[250.10,252.30] vol=2.2x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-04-05 09:45:00 | 249.58 | 250.10 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-04-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:00:00 | 259.00 | 257.72 | 0.00 | ORB-long ORB[256.00,258.45] vol=3.3x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-04-09 10:05:00 | 258.12 | 257.78 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-04-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 10:20:00 | 255.55 | 256.42 | 0.00 | ORB-short ORB[256.00,259.55] vol=2.0x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-04-22 11:05:00 | 256.58 | 256.33 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 09:30:00 | 258.60 | 259.95 | 0.00 | ORB-short ORB[259.05,261.80] vol=1.7x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-04-24 09:45:00 | 259.31 | 259.71 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-04-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 10:30:00 | 267.95 | 266.05 | 0.00 | ORB-long ORB[264.10,266.10] vol=3.6x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 10:40:00 | 269.56 | 266.71 | 0.00 | T1 1.5R @ 269.56 |
| Target hit | 2024-04-25 15:20:00 | 271.80 | 270.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2024-04-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:45:00 | 280.05 | 276.93 | 0.00 | ORB-long ORB[274.25,277.35] vol=4.7x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-04-26 10:00:00 | 278.63 | 277.84 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 11:00:00 | 81.45 | 2023-05-12 15:00:00 | 81.09 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-05-15 11:15:00 | 80.40 | 2023-05-15 12:05:00 | 80.59 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-16 09:45:00 | 81.70 | 2023-05-16 09:50:00 | 82.02 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-05-16 09:45:00 | 81.70 | 2023-05-16 10:15:00 | 81.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-19 09:30:00 | 80.35 | 2023-05-19 09:35:00 | 79.84 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2023-05-19 09:30:00 | 80.35 | 2023-05-19 10:15:00 | 80.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-24 09:30:00 | 80.55 | 2023-05-24 09:35:00 | 80.86 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-05-24 09:30:00 | 80.55 | 2023-05-24 10:20:00 | 80.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-25 11:00:00 | 80.70 | 2023-05-25 11:05:00 | 81.01 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-05-25 11:00:00 | 80.70 | 2023-05-25 11:15:00 | 80.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-30 09:30:00 | 83.90 | 2023-05-30 09:35:00 | 83.51 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2023-06-02 10:00:00 | 82.85 | 2023-06-02 10:20:00 | 83.22 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-06-02 10:00:00 | 82.85 | 2023-06-02 12:05:00 | 83.05 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2023-06-05 09:50:00 | 84.70 | 2023-06-05 11:45:00 | 85.18 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-06-05 09:50:00 | 84.70 | 2023-06-05 13:05:00 | 84.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-08 11:05:00 | 85.40 | 2023-06-08 11:15:00 | 85.75 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-06-08 11:05:00 | 85.40 | 2023-06-08 11:20:00 | 85.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-09 09:30:00 | 83.60 | 2023-06-09 09:35:00 | 83.13 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-06-09 09:30:00 | 83.60 | 2023-06-09 09:45:00 | 83.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-12 09:45:00 | 86.55 | 2023-06-12 09:55:00 | 86.15 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-06-15 11:15:00 | 84.50 | 2023-06-15 11:20:00 | 84.31 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-21 11:10:00 | 87.10 | 2023-06-21 11:20:00 | 87.32 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-06-22 09:55:00 | 85.75 | 2023-06-22 10:10:00 | 86.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-06-30 09:30:00 | 86.40 | 2023-06-30 09:40:00 | 86.11 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-07-03 10:50:00 | 88.85 | 2023-07-03 10:55:00 | 88.53 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-07-04 09:55:00 | 86.90 | 2023-07-04 10:05:00 | 87.26 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-07-05 09:35:00 | 88.75 | 2023-07-05 09:40:00 | 89.23 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-07-05 09:35:00 | 88.75 | 2023-07-05 15:20:00 | 93.15 | TARGET_HIT | 0.50 | 4.96% |
| SELL | retest1 | 2023-07-07 10:30:00 | 91.15 | 2023-07-07 10:35:00 | 90.70 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-07-07 10:30:00 | 91.15 | 2023-07-07 13:30:00 | 90.90 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2023-07-11 09:40:00 | 92.85 | 2023-07-11 11:00:00 | 93.40 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-07-11 09:40:00 | 92.85 | 2023-07-11 15:20:00 | 93.95 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2023-07-12 09:50:00 | 95.70 | 2023-07-12 09:55:00 | 95.41 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-17 10:30:00 | 93.15 | 2023-07-17 11:15:00 | 92.88 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-18 09:50:00 | 93.65 | 2023-07-18 09:55:00 | 94.16 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2023-07-18 09:50:00 | 93.65 | 2023-07-18 11:35:00 | 94.55 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2023-07-19 11:15:00 | 94.40 | 2023-07-19 11:25:00 | 94.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-21 09:30:00 | 95.95 | 2023-07-21 09:35:00 | 96.33 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-07-21 09:30:00 | 95.95 | 2023-07-21 09:35:00 | 95.85 | TARGET_HIT | 0.50 | -0.10% |
| BUY | retest1 | 2023-07-24 10:50:00 | 96.20 | 2023-07-24 11:25:00 | 95.90 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-26 09:45:00 | 99.15 | 2023-07-26 10:10:00 | 98.78 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-07-27 09:50:00 | 102.70 | 2023-07-27 10:05:00 | 103.25 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-27 09:50:00 | 102.70 | 2023-07-27 11:00:00 | 102.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-02 10:50:00 | 103.05 | 2023-08-02 11:00:00 | 102.54 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-08-02 10:50:00 | 103.05 | 2023-08-02 15:20:00 | 100.55 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2023-08-08 09:45:00 | 98.65 | 2023-08-08 09:50:00 | 99.24 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2023-08-08 09:45:00 | 98.65 | 2023-08-08 10:45:00 | 99.15 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-08-11 09:45:00 | 101.00 | 2023-08-11 10:25:00 | 101.47 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-08-11 09:45:00 | 101.00 | 2023-08-11 11:10:00 | 101.50 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2023-08-16 10:00:00 | 100.25 | 2023-08-16 10:10:00 | 100.64 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-08-17 09:30:00 | 100.55 | 2023-08-17 09:40:00 | 101.06 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-08-17 09:30:00 | 100.55 | 2023-08-17 10:55:00 | 100.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-18 09:35:00 | 99.35 | 2023-08-18 09:45:00 | 99.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-08-22 10:05:00 | 103.20 | 2023-08-22 10:10:00 | 102.81 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-09-05 09:35:00 | 142.30 | 2023-09-05 09:40:00 | 141.36 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2023-09-07 09:30:00 | 136.75 | 2023-09-07 09:35:00 | 137.43 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-09-07 09:30:00 | 136.75 | 2023-09-07 11:30:00 | 137.05 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2023-09-15 11:15:00 | 130.05 | 2023-09-15 11:25:00 | 130.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-21 09:30:00 | 123.90 | 2023-09-21 09:35:00 | 123.42 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-09-28 09:50:00 | 128.30 | 2023-09-28 09:55:00 | 129.22 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2023-09-28 09:50:00 | 128.30 | 2023-09-28 10:00:00 | 128.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-03 10:30:00 | 132.10 | 2023-10-03 10:50:00 | 131.49 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2023-10-06 10:20:00 | 129.30 | 2023-10-06 10:25:00 | 128.90 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-10-13 09:55:00 | 128.45 | 2023-10-13 10:00:00 | 127.78 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-10-13 09:55:00 | 128.45 | 2023-10-13 10:25:00 | 128.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-16 11:00:00 | 129.00 | 2023-10-16 11:15:00 | 128.62 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-17 10:00:00 | 132.55 | 2023-10-17 10:15:00 | 133.24 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-10-17 10:00:00 | 132.55 | 2023-10-17 10:20:00 | 132.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-18 09:45:00 | 133.40 | 2023-10-18 10:45:00 | 132.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-10-26 09:40:00 | 113.90 | 2023-10-26 09:50:00 | 114.52 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2023-10-27 10:05:00 | 120.25 | 2023-10-27 10:10:00 | 119.74 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-11-01 09:40:00 | 121.60 | 2023-11-01 09:45:00 | 121.16 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-11-02 10:25:00 | 124.40 | 2023-11-02 12:00:00 | 125.14 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-11-02 10:25:00 | 124.40 | 2023-11-02 12:10:00 | 124.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-03 09:30:00 | 130.35 | 2023-11-03 09:40:00 | 131.31 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2023-11-03 09:30:00 | 130.35 | 2023-11-03 10:10:00 | 130.80 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2023-11-06 10:15:00 | 130.50 | 2023-11-06 10:50:00 | 129.93 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-11-08 09:45:00 | 128.95 | 2023-11-08 09:55:00 | 129.32 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-11-20 09:45:00 | 140.30 | 2023-11-20 10:25:00 | 140.91 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-12-06 10:40:00 | 178.55 | 2023-12-06 12:05:00 | 179.85 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2023-12-06 10:40:00 | 178.55 | 2023-12-06 12:20:00 | 178.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-11 09:40:00 | 180.40 | 2023-12-11 10:00:00 | 179.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2023-12-12 10:10:00 | 177.25 | 2023-12-12 10:30:00 | 177.88 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-13 10:55:00 | 181.35 | 2023-12-13 11:15:00 | 180.48 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-12-20 09:35:00 | 188.95 | 2023-12-20 09:40:00 | 190.26 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2023-12-20 09:35:00 | 188.95 | 2023-12-20 09:45:00 | 188.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 09:45:00 | 180.90 | 2023-12-26 10:35:00 | 179.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2023-12-28 09:40:00 | 184.45 | 2023-12-28 09:45:00 | 185.63 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-12-28 09:40:00 | 184.45 | 2023-12-28 10:00:00 | 187.30 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2024-01-05 09:50:00 | 197.10 | 2024-01-05 10:00:00 | 196.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-01-08 09:55:00 | 199.20 | 2024-01-08 10:00:00 | 198.08 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-01-20 09:35:00 | 220.20 | 2024-01-20 09:40:00 | 221.43 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-01-25 09:55:00 | 215.30 | 2024-01-25 10:00:00 | 214.21 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-01-31 11:10:00 | 226.90 | 2024-01-31 11:25:00 | 225.49 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-01-31 11:10:00 | 226.90 | 2024-01-31 12:25:00 | 226.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 09:30:00 | 232.05 | 2024-02-02 09:35:00 | 233.74 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-02-02 09:30:00 | 232.05 | 2024-02-02 10:50:00 | 232.55 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2024-02-08 11:00:00 | 230.55 | 2024-02-08 11:25:00 | 231.49 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-02-16 10:50:00 | 229.50 | 2024-02-16 11:25:00 | 228.67 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-02-21 10:40:00 | 225.75 | 2024-02-21 12:50:00 | 226.68 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-02-27 11:05:00 | 224.65 | 2024-02-27 11:15:00 | 225.39 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-07 10:55:00 | 259.45 | 2024-03-07 11:25:00 | 258.14 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-04-05 09:35:00 | 248.60 | 2024-04-05 09:45:00 | 249.58 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-04-09 10:00:00 | 259.00 | 2024-04-09 10:05:00 | 258.12 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-04-22 10:20:00 | 255.55 | 2024-04-22 11:05:00 | 256.58 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-04-24 09:30:00 | 258.60 | 2024-04-24 09:45:00 | 259.31 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-25 10:30:00 | 267.95 | 2024-04-25 10:40:00 | 269.56 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-04-25 10:30:00 | 267.95 | 2024-04-25 15:20:00 | 271.80 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2024-04-26 09:45:00 | 280.05 | 2024-04-26 10:00:00 | 278.63 | STOP_HIT | 1.00 | -0.51% |
