# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16813 bars)
- **Last close:** 141.24
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
| ENTRY1 | 82 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 13 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 111 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 69
- **Target hits / Stop hits / Partials:** 13 / 69 / 29
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 10.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 16 | 29.6% | 5 | 38 | 11 | 0.07% | 3.8% |
| BUY @ 2nd Alert (retest1) | 54 | 16 | 29.6% | 5 | 38 | 11 | 0.07% | 3.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 26 | 45.6% | 8 | 31 | 18 | 0.11% | 6.3% |
| SELL @ 2nd Alert (retest1) | 57 | 26 | 45.6% | 8 | 31 | 18 | 0.11% | 6.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 111 | 42 | 37.8% | 13 | 69 | 29 | 0.09% | 10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:45:00 | 103.66 | 102.64 | 0.00 | ORB-long ORB[100.90,102.30] vol=8.5x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-05-16 10:20:00 | 103.19 | 103.15 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:10:00 | 105.75 | 104.75 | 0.00 | ORB-long ORB[104.10,105.49] vol=2.0x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-05-19 10:25:00 | 105.31 | 105.05 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 11:00:00 | 100.97 | 101.52 | 0.00 | ORB-short ORB[101.20,102.11] vol=2.9x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 11:15:00 | 100.59 | 101.46 | 0.00 | T1 1.5R @ 100.59 |
| Target hit | 2025-05-22 15:05:00 | 100.84 | 100.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2025-05-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:00:00 | 100.07 | 100.53 | 0.00 | ORB-short ORB[100.35,101.32] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-05-27 10:15:00 | 100.33 | 100.50 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 100.12 | 100.77 | 0.00 | ORB-short ORB[100.75,101.56] vol=2.6x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 11:05:00 | 99.63 | 100.25 | 0.00 | T1 1.5R @ 99.63 |
| Stop hit — per-position SL triggered | 2025-05-29 15:05:00 | 100.12 | 100.13 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 106.75 | 105.96 | 0.00 | ORB-long ORB[105.26,106.54] vol=3.7x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-06-03 09:35:00 | 106.29 | 106.04 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 104.46 | 103.99 | 0.00 | ORB-long ORB[103.20,104.20] vol=2.8x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-06-04 09:35:00 | 104.11 | 103.98 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 105.21 | 104.64 | 0.00 | ORB-long ORB[103.91,104.93] vol=4.9x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 104.78 | 104.61 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:40:00 | 108.64 | 108.03 | 0.00 | ORB-long ORB[107.25,108.20] vol=2.1x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-06-09 10:55:00 | 108.13 | 108.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:40:00 | 110.20 | 108.55 | 0.00 | ORB-long ORB[106.92,108.29] vol=8.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-06-10 10:50:00 | 109.56 | 108.79 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:00:00 | 102.21 | 102.87 | 0.00 | ORB-short ORB[102.25,103.30] vol=1.9x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:35:00 | 101.71 | 102.65 | 0.00 | T1 1.5R @ 101.71 |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 102.21 | 101.88 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-23 09:30:00 | 102.18 | 102.64 | 0.00 | ORB-short ORB[102.25,103.60] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-06-23 11:45:00 | 102.70 | 102.31 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:10:00 | 114.19 | 115.01 | 0.00 | ORB-short ORB[115.31,117.00] vol=2.1x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-07-02 11:20:00 | 114.55 | 114.97 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:15:00 | 113.73 | 112.87 | 0.00 | ORB-long ORB[112.20,113.57] vol=1.8x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-07-04 11:10:00 | 113.34 | 113.22 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:50:00 | 103.45 | 102.84 | 0.00 | ORB-long ORB[101.73,103.10] vol=3.4x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 14:30:00 | 104.71 | 103.63 | 0.00 | T1 1.5R @ 104.71 |
| Target hit | 2025-08-11 15:20:00 | 104.60 | 103.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:15:00 | 104.70 | 105.12 | 0.00 | ORB-short ORB[104.90,105.52] vol=2.4x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:25:00 | 104.43 | 105.08 | 0.00 | T1 1.5R @ 104.43 |
| Target hit | 2025-08-13 15:20:00 | 103.65 | 104.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 104.90 | 105.22 | 0.00 | ORB-short ORB[104.94,105.55] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-08-18 09:40:00 | 105.22 | 105.18 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 11:05:00 | 105.10 | 104.68 | 0.00 | ORB-long ORB[104.25,104.95] vol=1.7x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 104.95 | 104.69 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 103.59 | 104.01 | 0.00 | ORB-short ORB[103.80,104.64] vol=1.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 14:55:00 | 103.20 | 103.59 | 0.00 | T1 1.5R @ 103.20 |
| Target hit | 2025-08-22 15:20:00 | 102.60 | 103.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:55:00 | 100.53 | 99.36 | 0.00 | ORB-long ORB[98.75,99.55] vol=3.9x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-08-29 12:00:00 | 100.20 | 99.55 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:00:00 | 101.95 | 101.28 | 0.00 | ORB-long ORB[100.22,100.92] vol=3.6x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:15:00 | 102.40 | 101.61 | 0.00 | T1 1.5R @ 102.40 |
| Target hit | 2025-09-02 13:15:00 | 102.26 | 102.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2025-09-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:25:00 | 101.96 | 101.13 | 0.00 | ORB-long ORB[100.24,101.45] vol=2.9x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 10:30:00 | 102.42 | 101.30 | 0.00 | T1 1.5R @ 102.42 |
| Stop hit — per-position SL triggered | 2025-09-10 10:40:00 | 101.96 | 101.37 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:35:00 | 103.21 | 102.36 | 0.00 | ORB-long ORB[101.33,102.55] vol=5.8x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-09-11 09:40:00 | 102.86 | 102.48 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:50:00 | 102.25 | 101.59 | 0.00 | ORB-long ORB[101.01,101.90] vol=1.8x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-09-15 09:55:00 | 101.98 | 101.62 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:20:00 | 103.40 | 102.66 | 0.00 | ORB-long ORB[102.15,102.70] vol=5.4x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:25:00 | 103.81 | 103.16 | 0.00 | T1 1.5R @ 103.81 |
| Stop hit — per-position SL triggered | 2025-09-17 10:45:00 | 103.40 | 103.33 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:55:00 | 105.13 | 104.61 | 0.00 | ORB-long ORB[104.07,105.00] vol=2.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-09-18 10:10:00 | 104.78 | 104.68 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:50:00 | 105.15 | 104.55 | 0.00 | ORB-long ORB[104.20,104.79] vol=2.3x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-09-19 09:55:00 | 104.81 | 104.62 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:30:00 | 105.98 | 105.54 | 0.00 | ORB-long ORB[104.70,105.80] vol=1.8x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-09-22 09:50:00 | 105.64 | 105.72 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 103.79 | 104.23 | 0.00 | ORB-short ORB[104.05,104.73] vol=1.5x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:45:00 | 103.31 | 104.06 | 0.00 | T1 1.5R @ 103.31 |
| Stop hit — per-position SL triggered | 2025-09-23 10:05:00 | 103.79 | 103.82 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:00:00 | 105.79 | 105.04 | 0.00 | ORB-long ORB[104.62,105.50] vol=1.5x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:20:00 | 106.37 | 105.83 | 0.00 | T1 1.5R @ 106.37 |
| Stop hit — per-position SL triggered | 2025-09-24 10:35:00 | 105.79 | 105.84 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 100.95 | 101.47 | 0.00 | ORB-short ORB[101.28,102.60] vol=3.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-09-26 09:45:00 | 101.27 | 101.32 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:25:00 | 101.80 | 100.82 | 0.00 | ORB-long ORB[99.55,100.53] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-09-29 10:55:00 | 101.48 | 100.95 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:50:00 | 104.40 | 105.12 | 0.00 | ORB-short ORB[104.93,105.90] vol=2.1x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-10-06 12:15:00 | 104.74 | 104.95 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 104.00 | 104.35 | 0.00 | ORB-short ORB[104.18,104.89] vol=2.5x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 104.23 | 104.26 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:50:00 | 103.10 | 103.61 | 0.00 | ORB-short ORB[103.45,104.40] vol=3.5x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-10-08 09:55:00 | 103.37 | 103.57 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:55:00 | 104.50 | 103.78 | 0.00 | ORB-long ORB[103.03,103.87] vol=3.3x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:05:00 | 104.97 | 104.23 | 0.00 | T1 1.5R @ 104.97 |
| Target hit | 2025-10-10 15:20:00 | 107.45 | 106.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-10-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:25:00 | 106.89 | 106.46 | 0.00 | ORB-long ORB[105.75,106.73] vol=1.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-10-27 10:30:00 | 106.61 | 106.50 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:10:00 | 107.33 | 108.01 | 0.00 | ORB-short ORB[108.12,109.35] vol=1.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-10-29 11:25:00 | 107.61 | 107.97 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:30:00 | 107.06 | 107.47 | 0.00 | ORB-short ORB[107.24,107.80] vol=1.9x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 09:40:00 | 106.69 | 107.33 | 0.00 | T1 1.5R @ 106.69 |
| Target hit | 2025-10-30 15:20:00 | 105.74 | 106.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:45:00 | 106.82 | 106.59 | 0.00 | ORB-long ORB[105.74,106.76] vol=1.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-10-31 09:50:00 | 106.53 | 106.59 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:05:00 | 106.23 | 106.84 | 0.00 | ORB-short ORB[106.80,108.01] vol=1.6x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:45:00 | 105.86 | 106.61 | 0.00 | T1 1.5R @ 105.86 |
| Stop hit — per-position SL triggered | 2025-11-04 11:00:00 | 106.23 | 106.59 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 104.70 | 105.17 | 0.00 | ORB-short ORB[105.15,105.79] vol=2.6x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:45:00 | 104.32 | 105.00 | 0.00 | T1 1.5R @ 104.32 |
| Target hit | 2025-11-06 15:20:00 | 102.72 | 103.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-11-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:05:00 | 107.50 | 107.26 | 0.00 | ORB-long ORB[106.73,107.40] vol=1.9x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 107.28 | 107.27 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:50:00 | 106.94 | 107.25 | 0.00 | ORB-short ORB[107.36,108.05] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-11-12 09:55:00 | 107.16 | 107.24 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 106.73 | 106.55 | 0.00 | ORB-long ORB[106.10,106.65] vol=2.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-11-14 09:45:00 | 106.47 | 106.57 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 10:45:00 | 108.57 | 107.85 | 0.00 | ORB-long ORB[107.50,108.33] vol=1.9x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-11-18 10:50:00 | 108.20 | 107.86 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:15:00 | 107.95 | 107.24 | 0.00 | ORB-long ORB[106.49,107.52] vol=5.5x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-11-19 10:25:00 | 107.57 | 107.32 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:55:00 | 107.65 | 108.31 | 0.00 | ORB-short ORB[108.45,109.90] vol=2.3x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 12:20:00 | 107.27 | 108.05 | 0.00 | T1 1.5R @ 107.27 |
| Stop hit — per-position SL triggered | 2025-11-20 13:45:00 | 107.65 | 107.73 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 107.18 | 106.40 | 0.00 | ORB-long ORB[105.40,106.90] vol=3.4x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-11-26 09:35:00 | 106.84 | 106.43 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 107.00 | 107.49 | 0.00 | ORB-short ORB[107.16,108.18] vol=2.6x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:45:00 | 106.55 | 107.38 | 0.00 | T1 1.5R @ 106.55 |
| Stop hit — per-position SL triggered | 2025-11-27 09:50:00 | 107.00 | 107.36 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:45:00 | 108.37 | 107.89 | 0.00 | ORB-long ORB[107.22,108.25] vol=2.8x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-12-01 09:50:00 | 107.99 | 107.89 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:35:00 | 102.99 | 103.26 | 0.00 | ORB-short ORB[103.11,103.91] vol=2.2x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:55:00 | 102.64 | 103.12 | 0.00 | T1 1.5R @ 102.64 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 102.99 | 103.13 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:50:00 | 102.04 | 102.36 | 0.00 | ORB-short ORB[102.20,103.48] vol=1.9x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:05:00 | 101.65 | 102.22 | 0.00 | T1 1.5R @ 101.65 |
| Stop hit — per-position SL triggered | 2025-12-08 10:25:00 | 102.04 | 102.14 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:10:00 | 102.60 | 102.15 | 0.00 | ORB-long ORB[101.40,102.37] vol=1.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-12-10 10:25:00 | 102.31 | 102.22 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:35:00 | 101.88 | 102.23 | 0.00 | ORB-short ORB[102.00,102.62] vol=1.7x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:55:00 | 101.53 | 102.04 | 0.00 | T1 1.5R @ 101.53 |
| Stop hit — per-position SL triggered | 2025-12-15 10:00:00 | 101.88 | 102.03 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:00:00 | 100.01 | 100.38 | 0.00 | ORB-short ORB[100.32,101.57] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-12-16 10:05:00 | 100.27 | 100.37 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:20:00 | 99.45 | 100.29 | 0.00 | ORB-short ORB[100.03,101.38] vol=2.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-12-17 10:25:00 | 99.77 | 100.27 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:05:00 | 98.95 | 99.69 | 0.00 | ORB-short ORB[99.70,100.36] vol=3.6x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-12-19 11:35:00 | 99.14 | 99.54 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 09:40:00 | 100.05 | 100.35 | 0.00 | ORB-short ORB[100.32,100.86] vol=2.4x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-12-23 09:45:00 | 100.24 | 100.33 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:45:00 | 99.65 | 100.05 | 0.00 | ORB-short ORB[99.93,100.47] vol=2.2x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 13:30:00 | 99.39 | 99.83 | 0.00 | T1 1.5R @ 99.39 |
| Target hit | 2025-12-24 15:20:00 | 99.34 | 99.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 98.89 | 99.11 | 0.00 | ORB-short ORB[98.91,99.86] vol=2.5x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 14:35:00 | 98.62 | 98.96 | 0.00 | T1 1.5R @ 98.62 |
| Target hit | 2025-12-26 15:20:00 | 98.50 | 98.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2025-12-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:45:00 | 98.09 | 98.28 | 0.00 | ORB-short ORB[98.20,99.00] vol=5.4x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:15:00 | 97.85 | 98.21 | 0.00 | T1 1.5R @ 97.85 |
| Stop hit — per-position SL triggered | 2025-12-29 11:40:00 | 98.09 | 98.19 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:25:00 | 98.22 | 97.81 | 0.00 | ORB-long ORB[97.40,98.14] vol=1.7x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-12-30 11:10:00 | 98.02 | 97.86 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:10:00 | 100.45 | 100.12 | 0.00 | ORB-long ORB[99.28,99.98] vol=1.9x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:30:00 | 100.78 | 100.20 | 0.00 | T1 1.5R @ 100.78 |
| Stop hit — per-position SL triggered | 2025-12-31 11:55:00 | 100.45 | 100.25 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:05:00 | 100.69 | 100.26 | 0.00 | ORB-long ORB[99.81,100.64] vol=2.4x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-01-01 10:15:00 | 100.44 | 100.30 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:50:00 | 105.13 | 104.61 | 0.00 | ORB-long ORB[103.53,104.95] vol=2.0x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-01-06 11:20:00 | 104.75 | 104.72 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:50:00 | 103.15 | 103.70 | 0.00 | ORB-short ORB[103.51,104.39] vol=1.7x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:00:00 | 102.80 | 103.62 | 0.00 | T1 1.5R @ 102.80 |
| Target hit | 2026-01-08 14:50:00 | 102.80 | 102.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — SELL (started 2026-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 09:30:00 | 102.52 | 103.18 | 0.00 | ORB-short ORB[102.86,103.60] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-01-16 09:35:00 | 102.83 | 103.11 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:00:00 | 102.16 | 102.47 | 0.00 | ORB-short ORB[102.20,103.00] vol=2.0x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-01-19 10:05:00 | 102.49 | 102.46 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 10:05:00 | 100.50 | 101.18 | 0.00 | ORB-short ORB[100.90,102.05] vol=2.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-02-02 10:10:00 | 100.96 | 101.14 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 10:15:00 | 103.66 | 103.34 | 0.00 | ORB-long ORB[102.50,103.47] vol=1.7x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-04 10:35:00 | 103.36 | 103.43 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:15:00 | 102.35 | 103.12 | 0.00 | ORB-short ORB[103.50,104.27] vol=1.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-06 11:35:00 | 102.65 | 103.06 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 103.98 | 104.52 | 0.00 | ORB-short ORB[104.45,105.90] vol=1.9x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-11 10:30:00 | 104.28 | 104.33 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 101.74 | 102.28 | 0.00 | ORB-short ORB[102.26,103.03] vol=1.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 102.10 | 102.15 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 103.50 | 103.11 | 0.00 | ORB-long ORB[102.50,103.45] vol=2.2x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:20:00 | 103.91 | 103.25 | 0.00 | T1 1.5R @ 103.91 |
| Target hit | 2026-02-17 14:45:00 | 104.47 | 104.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 76 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 103.80 | 103.45 | 0.00 | ORB-long ORB[102.50,103.77] vol=2.1x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-02-20 11:00:00 | 103.35 | 103.66 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 107.28 | 106.06 | 0.00 | ORB-long ORB[105.31,106.14] vol=3.7x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:55:00 | 107.97 | 106.56 | 0.00 | T1 1.5R @ 107.97 |
| Target hit | 2026-02-24 10:30:00 | 111.27 | 111.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 78 — BUY (started 2026-04-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 124.46 | 123.56 | 0.00 | ORB-long ORB[122.27,123.95] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 123.96 | 123.59 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 127.55 | 126.53 | 0.00 | ORB-long ORB[125.30,126.95] vol=2.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-04-15 09:50:00 | 126.87 | 126.59 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 127.92 | 128.07 | 0.00 | ORB-short ORB[128.24,129.60] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 128.47 | 128.10 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:10:00 | 134.19 | 133.41 | 0.00 | ORB-long ORB[132.72,134.00] vol=1.7x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:25:00 | 134.83 | 133.65 | 0.00 | T1 1.5R @ 134.83 |
| Stop hit — per-position SL triggered | 2026-04-22 11:35:00 | 134.19 | 133.68 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 131.85 | 131.32 | 0.00 | ORB-long ORB[130.01,131.70] vol=1.7x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:40:00 | 132.47 | 131.43 | 0.00 | T1 1.5R @ 132.47 |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 131.85 | 131.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 09:45:00 | 103.66 | 2025-05-16 10:20:00 | 103.19 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-05-19 10:10:00 | 105.75 | 2025-05-19 10:25:00 | 105.31 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-22 11:00:00 | 100.97 | 2025-05-22 11:15:00 | 100.59 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-22 11:00:00 | 100.97 | 2025-05-22 15:05:00 | 100.84 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-05-27 10:00:00 | 100.07 | 2025-05-27 10:15:00 | 100.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-05-29 09:35:00 | 100.12 | 2025-05-29 11:05:00 | 99.63 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-05-29 09:35:00 | 100.12 | 2025-05-29 15:05:00 | 100.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-03 09:30:00 | 106.75 | 2025-06-03 09:35:00 | 106.29 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-04 09:30:00 | 104.46 | 2025-06-04 09:35:00 | 104.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-06 10:05:00 | 105.21 | 2025-06-06 10:10:00 | 104.78 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-06-09 09:40:00 | 108.64 | 2025-06-09 10:55:00 | 108.13 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-06-10 10:40:00 | 110.20 | 2025-06-10 10:50:00 | 109.56 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-06-19 10:00:00 | 102.21 | 2025-06-19 10:35:00 | 101.71 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-06-19 10:00:00 | 102.21 | 2025-06-19 14:15:00 | 102.21 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-23 09:30:00 | 102.18 | 2025-06-23 11:45:00 | 102.70 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-07-02 11:10:00 | 114.19 | 2025-07-02 11:20:00 | 114.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-04 10:15:00 | 113.73 | 2025-07-04 11:10:00 | 113.34 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-11 09:50:00 | 103.45 | 2025-08-11 14:30:00 | 104.71 | PARTIAL | 0.50 | 1.21% |
| BUY | retest1 | 2025-08-11 09:50:00 | 103.45 | 2025-08-11 15:20:00 | 104.60 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2025-08-13 11:15:00 | 104.70 | 2025-08-13 11:25:00 | 104.43 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-08-13 11:15:00 | 104.70 | 2025-08-13 15:20:00 | 103.65 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2025-08-18 09:30:00 | 104.90 | 2025-08-18 09:40:00 | 105.22 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-21 11:05:00 | 105.10 | 2025-08-21 11:15:00 | 104.95 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-08-22 09:30:00 | 103.59 | 2025-08-22 14:55:00 | 103.20 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-22 09:30:00 | 103.59 | 2025-08-22 15:20:00 | 102.60 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2025-08-29 10:55:00 | 100.53 | 2025-08-29 12:00:00 | 100.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-02 10:00:00 | 101.95 | 2025-09-02 10:15:00 | 102.40 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-02 10:00:00 | 101.95 | 2025-09-02 13:15:00 | 102.26 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-10 10:25:00 | 101.96 | 2025-09-10 10:30:00 | 102.42 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-10 10:25:00 | 101.96 | 2025-09-10 10:40:00 | 101.96 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 09:35:00 | 103.21 | 2025-09-11 09:40:00 | 102.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-15 09:50:00 | 102.25 | 2025-09-15 09:55:00 | 101.98 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-17 10:20:00 | 103.40 | 2025-09-17 10:25:00 | 103.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-17 10:20:00 | 103.40 | 2025-09-17 10:45:00 | 103.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 09:55:00 | 105.13 | 2025-09-18 10:10:00 | 104.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-19 09:50:00 | 105.15 | 2025-09-19 09:55:00 | 104.81 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-22 09:30:00 | 105.98 | 2025-09-22 09:50:00 | 105.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-23 09:30:00 | 103.79 | 2025-09-23 09:45:00 | 103.31 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-09-23 09:30:00 | 103.79 | 2025-09-23 10:05:00 | 103.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-24 10:00:00 | 105.79 | 2025-09-24 10:20:00 | 106.37 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-09-24 10:00:00 | 105.79 | 2025-09-24 10:35:00 | 105.79 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-26 09:30:00 | 100.95 | 2025-09-26 09:45:00 | 101.27 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-29 10:25:00 | 101.80 | 2025-09-29 10:55:00 | 101.48 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-06 10:50:00 | 104.40 | 2025-10-06 12:15:00 | 104.74 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-07 11:10:00 | 104.00 | 2025-10-07 12:15:00 | 104.23 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-08 09:50:00 | 103.10 | 2025-10-08 09:55:00 | 103.37 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-10 09:55:00 | 104.50 | 2025-10-10 10:05:00 | 104.97 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-10-10 09:55:00 | 104.50 | 2025-10-10 15:20:00 | 107.45 | TARGET_HIT | 0.50 | 2.82% |
| BUY | retest1 | 2025-10-27 10:25:00 | 106.89 | 2025-10-27 10:30:00 | 106.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-29 11:10:00 | 107.33 | 2025-10-29 11:25:00 | 107.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-30 09:30:00 | 107.06 | 2025-10-30 09:40:00 | 106.69 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-30 09:30:00 | 107.06 | 2025-10-30 15:20:00 | 105.74 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2025-10-31 09:45:00 | 106.82 | 2025-10-31 09:50:00 | 106.53 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-04 10:05:00 | 106.23 | 2025-11-04 10:45:00 | 105.86 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-04 10:05:00 | 106.23 | 2025-11-04 11:00:00 | 106.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 09:30:00 | 104.70 | 2025-11-06 09:45:00 | 104.32 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-06 09:30:00 | 104.70 | 2025-11-06 15:20:00 | 102.72 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2025-11-11 10:05:00 | 107.50 | 2025-11-11 10:15:00 | 107.28 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-12 09:50:00 | 106.94 | 2025-11-12 09:55:00 | 107.16 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-14 09:30:00 | 106.73 | 2025-11-14 09:45:00 | 106.47 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-18 10:45:00 | 108.57 | 2025-11-18 10:50:00 | 108.20 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-19 10:15:00 | 107.95 | 2025-11-19 10:25:00 | 107.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-20 10:55:00 | 107.65 | 2025-11-20 12:20:00 | 107.27 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-20 10:55:00 | 107.65 | 2025-11-20 13:45:00 | 107.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 09:30:00 | 107.18 | 2025-11-26 09:35:00 | 106.84 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-27 09:30:00 | 107.00 | 2025-11-27 09:45:00 | 106.55 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-11-27 09:30:00 | 107.00 | 2025-11-27 09:50:00 | 107.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 09:45:00 | 108.37 | 2025-12-01 09:50:00 | 107.99 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-05 09:35:00 | 102.99 | 2025-12-05 09:55:00 | 102.64 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-05 09:35:00 | 102.99 | 2025-12-05 10:00:00 | 102.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 09:50:00 | 102.04 | 2025-12-08 10:05:00 | 101.65 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-08 09:50:00 | 102.04 | 2025-12-08 10:25:00 | 102.04 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 10:10:00 | 102.60 | 2025-12-10 10:25:00 | 102.31 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-15 09:35:00 | 101.88 | 2025-12-15 09:55:00 | 101.53 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-15 09:35:00 | 101.88 | 2025-12-15 10:00:00 | 101.88 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 10:00:00 | 100.01 | 2025-12-16 10:05:00 | 100.27 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-17 10:20:00 | 99.45 | 2025-12-17 10:25:00 | 99.77 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-19 11:05:00 | 98.95 | 2025-12-19 11:35:00 | 99.14 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-23 09:40:00 | 100.05 | 2025-12-23 09:45:00 | 100.24 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-24 10:45:00 | 99.65 | 2025-12-24 13:30:00 | 99.39 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-24 10:45:00 | 99.65 | 2025-12-24 15:20:00 | 99.34 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-26 11:10:00 | 98.89 | 2025-12-26 14:35:00 | 98.62 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-26 11:10:00 | 98.89 | 2025-12-26 15:20:00 | 98.50 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-29 10:45:00 | 98.09 | 2025-12-29 11:15:00 | 97.85 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-29 10:45:00 | 98.09 | 2025-12-29 11:40:00 | 98.09 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:25:00 | 98.22 | 2025-12-30 11:10:00 | 98.02 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-31 11:10:00 | 100.45 | 2025-12-31 11:30:00 | 100.78 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-12-31 11:10:00 | 100.45 | 2025-12-31 11:55:00 | 100.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-01 10:05:00 | 100.69 | 2026-01-01 10:15:00 | 100.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-06 10:50:00 | 105.13 | 2026-01-06 11:20:00 | 104.75 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-08 10:50:00 | 103.15 | 2026-01-08 11:00:00 | 102.80 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-08 10:50:00 | 103.15 | 2026-01-08 14:50:00 | 102.80 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-16 09:30:00 | 102.52 | 2026-01-16 09:35:00 | 102.83 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-19 10:00:00 | 102.16 | 2026-01-19 10:05:00 | 102.49 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-02 10:05:00 | 100.50 | 2026-02-02 10:10:00 | 100.96 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-04 10:15:00 | 103.66 | 2026-02-04 10:35:00 | 103.36 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-06 11:15:00 | 102.35 | 2026-02-06 11:35:00 | 102.65 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-11 09:40:00 | 103.98 | 2026-02-11 10:30:00 | 104.28 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-13 09:30:00 | 101.74 | 2026-02-13 09:40:00 | 102.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-17 10:10:00 | 103.50 | 2026-02-17 10:20:00 | 103.91 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-17 10:10:00 | 103.50 | 2026-02-17 14:45:00 | 104.47 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2026-02-20 09:35:00 | 103.80 | 2026-02-20 11:00:00 | 103.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-24 09:45:00 | 107.28 | 2026-02-24 09:55:00 | 107.97 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-24 09:45:00 | 107.28 | 2026-02-24 10:30:00 | 111.27 | TARGET_HIT | 0.50 | 3.72% |
| BUY | retest1 | 2026-04-10 10:00:00 | 124.46 | 2026-04-10 10:05:00 | 123.96 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-15 09:40:00 | 127.55 | 2026-04-15 09:50:00 | 126.87 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-04-16 10:15:00 | 127.92 | 2026-04-16 10:25:00 | 128.47 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-22 11:10:00 | 134.19 | 2026-04-22 11:25:00 | 134.83 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-22 11:10:00 | 134.19 | 2026-04-22 11:35:00 | 134.19 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:10:00 | 131.85 | 2026-05-04 11:40:00 | 132.47 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-04 11:10:00 | 131.85 | 2026-05-04 12:15:00 | 131.85 | STOP_HIT | 0.50 | 0.00% |
