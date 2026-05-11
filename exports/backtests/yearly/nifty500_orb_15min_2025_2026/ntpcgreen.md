# NTPC Green Energy Ltd. (NTPCGREEN)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 107.55
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
| ENTRY1 | 66 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 18 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 97 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 48
- **Target hits / Stop hits / Partials:** 18 / 48 / 31
- **Avg / median % per leg:** 0.36% / 0.07%
- **Sum % (uncompounded):** 35.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 20 | 48.8% | 8 | 21 | 12 | 0.61% | 25.1% |
| BUY @ 2nd Alert (retest1) | 41 | 20 | 48.8% | 8 | 21 | 12 | 0.61% | 25.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 29 | 51.8% | 10 | 27 | 19 | 0.18% | 9.9% |
| SELL @ 2nd Alert (retest1) | 56 | 29 | 51.8% | 10 | 27 | 19 | 0.18% | 9.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 97 | 49 | 50.5% | 18 | 48 | 31 | 0.36% | 35.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:50:00 | 102.60 | 102.44 | 0.00 | ORB-long ORB[101.90,102.56] vol=3.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-05-16 09:55:00 | 102.34 | 102.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:15:00 | 108.99 | 109.77 | 0.00 | ORB-short ORB[109.39,110.30] vol=2.1x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:35:00 | 108.68 | 109.70 | 0.00 | T1 1.5R @ 108.68 |
| Target hit | 2025-06-03 15:20:00 | 107.55 | 108.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-06-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:45:00 | 107.10 | 107.28 | 0.00 | ORB-short ORB[107.34,108.38] vol=2.3x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:20:00 | 106.75 | 107.25 | 0.00 | T1 1.5R @ 106.75 |
| Stop hit — per-position SL triggered | 2025-06-04 14:05:00 | 107.10 | 107.16 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:40:00 | 108.22 | 108.56 | 0.00 | ORB-short ORB[108.40,109.61] vol=2.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-06-06 11:05:00 | 108.58 | 108.53 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:30:00 | 105.64 | 104.39 | 0.00 | ORB-long ORB[103.44,104.74] vol=2.5x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:40:00 | 106.39 | 105.95 | 0.00 | T1 1.5R @ 106.39 |
| Target hit | 2025-06-20 15:20:00 | 109.98 | 109.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 106.76 | 107.27 | 0.00 | ORB-short ORB[107.25,108.35] vol=1.7x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-06-26 11:40:00 | 106.94 | 107.23 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 09:45:00 | 106.25 | 106.84 | 0.00 | ORB-short ORB[106.41,107.85] vol=1.5x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-06-27 11:15:00 | 106.53 | 106.66 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 11:15:00 | 105.86 | 105.08 | 0.00 | ORB-long ORB[104.80,105.84] vol=5.2x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 12:40:00 | 106.33 | 105.41 | 0.00 | T1 1.5R @ 106.33 |
| Target hit | 2025-07-01 15:20:00 | 107.54 | 106.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:45:00 | 106.76 | 106.35 | 0.00 | ORB-long ORB[105.73,106.50] vol=2.4x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-07-04 11:40:00 | 106.56 | 106.45 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:30:00 | 108.28 | 107.88 | 0.00 | ORB-long ORB[107.00,107.64] vol=8.7x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:40:00 | 108.60 | 108.08 | 0.00 | T1 1.5R @ 108.60 |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 108.28 | 108.25 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:55:00 | 109.67 | 108.93 | 0.00 | ORB-long ORB[108.01,109.44] vol=2.2x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:00:00 | 110.24 | 109.26 | 0.00 | T1 1.5R @ 110.24 |
| Target hit | 2025-07-11 15:20:00 | 112.29 | 111.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:05:00 | 107.86 | 108.46 | 0.00 | ORB-short ORB[108.30,109.00] vol=2.0x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-07-23 10:30:00 | 108.15 | 108.27 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:15:00 | 105.68 | 106.20 | 0.00 | ORB-short ORB[105.78,106.80] vol=2.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-07-25 11:20:00 | 105.94 | 106.08 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:55:00 | 107.84 | 106.63 | 0.00 | ORB-long ORB[105.45,106.78] vol=2.8x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-07-28 10:05:00 | 107.42 | 106.83 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:35:00 | 104.40 | 103.90 | 0.00 | ORB-long ORB[103.22,104.10] vol=1.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-08-07 09:40:00 | 104.10 | 103.92 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:05:00 | 102.37 | 101.60 | 0.00 | ORB-long ORB[101.37,102.23] vol=1.9x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-08-11 11:10:00 | 102.08 | 101.62 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:15:00 | 103.53 | 102.80 | 0.00 | ORB-long ORB[101.70,102.95] vol=3.9x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:25:00 | 103.96 | 102.99 | 0.00 | T1 1.5R @ 103.96 |
| Stop hit — per-position SL triggered | 2025-08-12 10:30:00 | 103.53 | 103.02 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:55:00 | 102.49 | 102.85 | 0.00 | ORB-short ORB[102.70,103.45] vol=2.8x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:40:00 | 102.19 | 102.73 | 0.00 | T1 1.5R @ 102.19 |
| Target hit | 2025-08-13 15:20:00 | 100.55 | 101.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-08-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 11:00:00 | 101.55 | 101.79 | 0.00 | ORB-short ORB[101.59,102.28] vol=1.9x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 11:45:00 | 101.23 | 101.73 | 0.00 | T1 1.5R @ 101.23 |
| Stop hit — per-position SL triggered | 2025-08-18 13:30:00 | 101.55 | 101.63 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:50:00 | 103.17 | 103.36 | 0.00 | ORB-short ORB[103.50,104.52] vol=2.4x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:30:00 | 102.86 | 103.20 | 0.00 | T1 1.5R @ 102.86 |
| Target hit | 2025-08-25 14:50:00 | 103.00 | 102.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — BUY (started 2025-09-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:05:00 | 103.60 | 102.89 | 0.00 | ORB-long ORB[102.32,103.25] vol=2.6x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-09-01 10:20:00 | 103.34 | 103.01 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:25:00 | 105.12 | 104.68 | 0.00 | ORB-long ORB[104.01,104.83] vol=3.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-09-02 10:35:00 | 104.86 | 104.76 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 10:40:00 | 104.06 | 104.98 | 0.00 | ORB-short ORB[104.95,105.95] vol=1.8x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:50:00 | 103.62 | 104.77 | 0.00 | T1 1.5R @ 103.62 |
| Target hit | 2025-09-08 15:20:00 | 102.81 | 103.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-09-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:00:00 | 104.15 | 103.82 | 0.00 | ORB-long ORB[103.39,104.05] vol=1.8x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-09-10 10:20:00 | 103.92 | 103.85 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:50:00 | 104.60 | 104.07 | 0.00 | ORB-long ORB[103.10,104.12] vol=5.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-09-11 09:55:00 | 104.35 | 104.12 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:45:00 | 104.40 | 103.99 | 0.00 | ORB-long ORB[103.00,104.32] vol=3.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 09:50:00 | 104.80 | 104.30 | 0.00 | T1 1.5R @ 104.80 |
| Target hit | 2025-09-15 11:40:00 | 104.52 | 104.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2025-09-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:10:00 | 104.56 | 105.13 | 0.00 | ORB-short ORB[104.90,106.00] vol=2.7x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-09-17 11:50:00 | 104.75 | 105.02 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:15:00 | 103.55 | 103.89 | 0.00 | ORB-short ORB[103.80,104.50] vol=2.5x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-09-22 12:25:00 | 103.72 | 103.82 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 103.01 | 103.45 | 0.00 | ORB-short ORB[103.20,103.94] vol=2.1x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-09-23 09:35:00 | 103.22 | 103.44 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 102.79 | 103.22 | 0.00 | ORB-short ORB[102.90,104.20] vol=4.4x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-09-24 12:55:00 | 103.00 | 103.13 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 100.14 | 100.80 | 0.00 | ORB-short ORB[100.60,101.88] vol=2.8x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-09-26 09:35:00 | 100.41 | 100.76 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:50:00 | 98.56 | 98.30 | 0.00 | ORB-long ORB[97.80,98.49] vol=2.5x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-10-07 11:20:00 | 98.33 | 98.45 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:40:00 | 97.89 | 98.17 | 0.00 | ORB-short ORB[98.05,98.77] vol=2.5x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:20:00 | 97.59 | 98.02 | 0.00 | T1 1.5R @ 97.59 |
| Stop hit — per-position SL triggered | 2025-10-09 10:45:00 | 97.89 | 97.96 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 97.35 | 98.17 | 0.00 | ORB-short ORB[98.52,99.00] vol=1.9x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-10-14 11:55:00 | 97.52 | 98.08 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:30:00 | 99.61 | 98.92 | 0.00 | ORB-long ORB[98.11,99.10] vol=1.7x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-10-15 12:45:00 | 99.36 | 99.30 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 11:10:00 | 99.80 | 99.95 | 0.00 | ORB-short ORB[99.90,100.47] vol=1.9x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 99.92 | 99.95 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 11:00:00 | 101.46 | 101.07 | 0.00 | ORB-long ORB[100.53,101.30] vol=3.6x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:05:00 | 101.75 | 101.31 | 0.00 | T1 1.5R @ 101.75 |
| Stop hit — per-position SL triggered | 2025-10-24 11:30:00 | 101.46 | 101.35 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:55:00 | 102.45 | 102.03 | 0.00 | ORB-long ORB[101.50,102.00] vol=6.0x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:00:00 | 102.76 | 102.32 | 0.00 | T1 1.5R @ 102.76 |
| Target hit | 2025-10-29 15:20:00 | 105.08 | 104.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2025-11-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:10:00 | 102.02 | 102.41 | 0.00 | ORB-short ORB[102.62,103.00] vol=2.8x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:15:00 | 101.84 | 102.37 | 0.00 | T1 1.5R @ 101.84 |
| Stop hit — per-position SL triggered | 2025-11-04 11:20:00 | 102.02 | 102.33 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:40:00 | 98.54 | 98.72 | 0.00 | ORB-short ORB[98.61,99.39] vol=1.8x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-11-07 09:50:00 | 98.74 | 98.71 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 11:00:00 | 98.47 | 99.02 | 0.00 | ORB-short ORB[98.90,99.50] vol=3.1x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 13:05:00 | 98.19 | 98.75 | 0.00 | T1 1.5R @ 98.19 |
| Target hit | 2025-11-10 15:20:00 | 97.80 | 98.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-11-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:45:00 | 98.76 | 98.92 | 0.00 | ORB-short ORB[98.77,99.20] vol=1.6x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-11-14 10:55:00 | 98.93 | 98.92 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 11:10:00 | 98.62 | 98.62 | 0.00 | ORB-short ORB[98.70,99.28] vol=1.8x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 12:45:00 | 98.42 | 98.59 | 0.00 | T1 1.5R @ 98.42 |
| Target hit | 2025-11-18 15:20:00 | 98.29 | 98.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-11-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:30:00 | 98.20 | 98.28 | 0.00 | ORB-short ORB[98.26,99.11] vol=8.4x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:35:00 | 97.97 | 98.24 | 0.00 | T1 1.5R @ 97.97 |
| Target hit | 2025-11-20 15:20:00 | 97.48 | 97.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 09:30:00 | 96.30 | 96.65 | 0.00 | ORB-short ORB[96.35,97.25] vol=1.6x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:40:00 | 96.07 | 96.56 | 0.00 | T1 1.5R @ 96.07 |
| Target hit | 2025-11-24 15:20:00 | 95.45 | 95.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2025-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:30:00 | 93.85 | 94.28 | 0.00 | ORB-short ORB[94.00,95.24] vol=2.4x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-11-25 09:55:00 | 94.10 | 94.17 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:30:00 | 93.60 | 93.75 | 0.00 | ORB-short ORB[93.71,94.33] vol=1.6x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 12:55:00 | 93.39 | 93.61 | 0.00 | T1 1.5R @ 93.39 |
| Target hit | 2025-12-02 15:20:00 | 92.93 | 93.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:50:00 | 90.76 | 91.38 | 0.00 | ORB-short ORB[91.02,91.71] vol=2.4x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:30:00 | 90.44 | 91.27 | 0.00 | T1 1.5R @ 90.44 |
| Stop hit — per-position SL triggered | 2025-12-10 15:00:00 | 90.76 | 91.02 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:25:00 | 89.69 | 89.91 | 0.00 | ORB-short ORB[89.76,90.24] vol=1.9x ATR=0.13 |
| Stop hit — per-position SL triggered | 2025-12-19 11:20:00 | 89.82 | 89.85 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:40:00 | 90.55 | 90.19 | 0.00 | ORB-long ORB[89.96,90.40] vol=2.0x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:05:00 | 90.79 | 90.30 | 0.00 | T1 1.5R @ 90.79 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 90.55 | 90.33 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:45:00 | 93.37 | 92.43 | 0.00 | ORB-long ORB[91.90,92.85] vol=2.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-12-29 09:50:00 | 93.11 | 92.50 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:15:00 | 93.98 | 94.63 | 0.00 | ORB-short ORB[94.25,95.60] vol=3.1x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:50:00 | 93.66 | 94.53 | 0.00 | T1 1.5R @ 93.66 |
| Stop hit — per-position SL triggered | 2025-12-30 12:00:00 | 93.98 | 94.51 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 94.16 | 93.77 | 0.00 | ORB-long ORB[93.41,94.11] vol=2.0x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-12-31 11:20:00 | 93.92 | 93.79 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:50:00 | 95.55 | 94.97 | 0.00 | ORB-long ORB[94.42,95.00] vol=8.2x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-01-01 11:00:00 | 95.32 | 95.05 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:05:00 | 96.19 | 95.65 | 0.00 | ORB-long ORB[94.83,96.01] vol=5.4x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 12:30:00 | 96.58 | 96.16 | 0.00 | T1 1.5R @ 96.58 |
| Target hit | 2026-01-02 15:20:00 | 97.04 | 96.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 11:15:00 | 94.27 | 93.54 | 0.00 | ORB-long ORB[93.02,93.84] vol=4.3x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-01-07 11:30:00 | 94.08 | 93.61 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 11:15:00 | 91.41 | 91.83 | 0.00 | ORB-short ORB[91.50,92.25] vol=4.0x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-01-14 11:30:00 | 91.68 | 91.81 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-02-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:05:00 | 89.23 | 88.55 | 0.00 | ORB-long ORB[88.00,88.50] vol=4.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 13:25:00 | 89.59 | 88.98 | 0.00 | T1 1.5R @ 89.59 |
| Target hit | 2026-02-16 15:20:00 | 89.51 | 89.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 89.48 | 89.63 | 0.00 | ORB-short ORB[89.50,90.15] vol=1.6x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:00:00 | 89.20 | 89.53 | 0.00 | T1 1.5R @ 89.20 |
| Stop hit — per-position SL triggered | 2026-02-18 10:20:00 | 89.48 | 89.48 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 89.25 | 89.29 | 0.00 | ORB-short ORB[89.31,89.73] vol=7.7x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:40:00 | 88.98 | 89.23 | 0.00 | T1 1.5R @ 88.98 |
| Target hit | 2026-02-19 13:35:00 | 89.19 | 89.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 89.34 | 89.53 | 0.00 | ORB-short ORB[89.45,90.00] vol=2.1x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-02-24 10:10:00 | 89.57 | 89.49 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 90.56 | 90.13 | 0.00 | ORB-long ORB[89.50,90.40] vol=4.1x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 90.37 | 90.19 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 89.61 | 89.67 | 0.00 | ORB-short ORB[89.65,90.23] vol=5.3x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 12:05:00 | 89.38 | 89.65 | 0.00 | T1 1.5R @ 89.38 |
| Stop hit — per-position SL triggered | 2026-02-27 12:45:00 | 89.61 | 89.62 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 88.09 | 87.56 | 0.00 | ORB-long ORB[86.65,87.88] vol=2.3x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-03-11 10:35:00 | 87.79 | 87.58 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 86.90 | 86.13 | 0.00 | ORB-long ORB[85.23,86.51] vol=2.2x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:40:00 | 87.49 | 86.52 | 0.00 | T1 1.5R @ 87.49 |
| Target hit | 2026-03-12 14:40:00 | 97.71 | 98.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 109.48 | 111.01 | 0.00 | ORB-short ORB[110.26,111.75] vol=2.7x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 15:05:00 | 108.70 | 109.76 | 0.00 | T1 1.5R @ 108.70 |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 109.48 | 109.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 09:50:00 | 102.60 | 2025-05-16 09:55:00 | 102.34 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-03 11:15:00 | 108.99 | 2025-06-03 11:35:00 | 108.68 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-03 11:15:00 | 108.99 | 2025-06-03 15:20:00 | 107.55 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2025-06-04 10:45:00 | 107.10 | 2025-06-04 11:20:00 | 106.75 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-06-04 10:45:00 | 107.10 | 2025-06-04 14:05:00 | 107.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 10:40:00 | 108.22 | 2025-06-06 11:05:00 | 108.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-20 09:30:00 | 105.64 | 2025-06-20 09:40:00 | 106.39 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-06-20 09:30:00 | 105.64 | 2025-06-20 15:20:00 | 109.98 | TARGET_HIT | 0.50 | 4.11% |
| SELL | retest1 | 2025-06-26 11:15:00 | 106.76 | 2025-06-26 11:40:00 | 106.94 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-06-27 09:45:00 | 106.25 | 2025-06-27 11:15:00 | 106.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-01 11:15:00 | 105.86 | 2025-07-01 12:40:00 | 106.33 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-07-01 11:15:00 | 105.86 | 2025-07-01 15:20:00 | 107.54 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2025-07-04 10:45:00 | 106.76 | 2025-07-04 11:40:00 | 106.56 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-10 09:30:00 | 108.28 | 2025-07-10 09:40:00 | 108.60 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-07-10 09:30:00 | 108.28 | 2025-07-10 10:15:00 | 108.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-11 09:55:00 | 109.67 | 2025-07-11 10:00:00 | 110.24 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-07-11 09:55:00 | 109.67 | 2025-07-11 15:20:00 | 112.29 | TARGET_HIT | 0.50 | 2.39% |
| SELL | retest1 | 2025-07-23 10:05:00 | 107.86 | 2025-07-23 10:30:00 | 108.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-25 10:15:00 | 105.68 | 2025-07-25 11:20:00 | 105.94 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-28 09:55:00 | 107.84 | 2025-07-28 10:05:00 | 107.42 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-08-07 09:35:00 | 104.40 | 2025-08-07 09:40:00 | 104.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-11 11:05:00 | 102.37 | 2025-08-11 11:10:00 | 102.08 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-12 10:15:00 | 103.53 | 2025-08-12 10:25:00 | 103.96 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-08-12 10:15:00 | 103.53 | 2025-08-12 10:30:00 | 103.53 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-13 10:55:00 | 102.49 | 2025-08-13 11:40:00 | 102.19 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-08-13 10:55:00 | 102.49 | 2025-08-13 15:20:00 | 100.55 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2025-08-18 11:00:00 | 101.55 | 2025-08-18 11:45:00 | 101.23 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-18 11:00:00 | 101.55 | 2025-08-18 13:30:00 | 101.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-25 10:50:00 | 103.17 | 2025-08-25 11:30:00 | 102.86 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-08-25 10:50:00 | 103.17 | 2025-08-25 14:50:00 | 103.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-09-01 10:05:00 | 103.60 | 2025-09-01 10:20:00 | 103.34 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-02 10:25:00 | 105.12 | 2025-09-02 10:35:00 | 104.86 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-08 10:40:00 | 104.06 | 2025-09-08 10:50:00 | 103.62 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-08 10:40:00 | 104.06 | 2025-09-08 15:20:00 | 102.81 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2025-09-10 10:00:00 | 104.15 | 2025-09-10 10:20:00 | 103.92 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-11 09:50:00 | 104.60 | 2025-09-11 09:55:00 | 104.35 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-15 09:45:00 | 104.40 | 2025-09-15 09:50:00 | 104.80 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-15 09:45:00 | 104.40 | 2025-09-15 11:40:00 | 104.52 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-09-17 11:10:00 | 104.56 | 2025-09-17 11:50:00 | 104.75 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-22 11:15:00 | 103.55 | 2025-09-22 12:25:00 | 103.72 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-09-23 09:30:00 | 103.01 | 2025-09-23 09:35:00 | 103.22 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-24 11:15:00 | 102.79 | 2025-09-24 12:55:00 | 103.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-26 09:30:00 | 100.14 | 2025-09-26 09:35:00 | 100.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-07 09:50:00 | 98.56 | 2025-10-07 11:20:00 | 98.33 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-09 09:40:00 | 97.89 | 2025-10-09 10:20:00 | 97.59 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-09 09:40:00 | 97.89 | 2025-10-09 10:45:00 | 97.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 11:15:00 | 97.35 | 2025-10-14 11:55:00 | 97.52 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-15 09:30:00 | 99.61 | 2025-10-15 12:45:00 | 99.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-23 11:10:00 | 99.80 | 2025-10-23 11:15:00 | 99.92 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-10-24 11:00:00 | 101.46 | 2025-10-24 11:05:00 | 101.75 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-24 11:00:00 | 101.46 | 2025-10-24 11:30:00 | 101.46 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 09:55:00 | 102.45 | 2025-10-29 10:00:00 | 102.76 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-29 09:55:00 | 102.45 | 2025-10-29 15:20:00 | 105.08 | TARGET_HIT | 0.50 | 2.57% |
| SELL | retest1 | 2025-11-04 11:10:00 | 102.02 | 2025-11-04 11:15:00 | 101.84 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-11-04 11:10:00 | 102.02 | 2025-11-04 11:20:00 | 102.02 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-07 09:40:00 | 98.54 | 2025-11-07 09:50:00 | 98.74 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-10 11:00:00 | 98.47 | 2025-11-10 13:05:00 | 98.19 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-10 11:00:00 | 98.47 | 2025-11-10 15:20:00 | 97.80 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2025-11-14 10:45:00 | 98.76 | 2025-11-14 10:55:00 | 98.93 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-18 11:10:00 | 98.62 | 2025-11-18 12:45:00 | 98.42 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-11-18 11:10:00 | 98.62 | 2025-11-18 15:20:00 | 98.29 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-20 10:30:00 | 98.20 | 2025-11-20 11:35:00 | 97.97 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-11-20 10:30:00 | 98.20 | 2025-11-20 15:20:00 | 97.48 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-11-24 09:30:00 | 96.30 | 2025-11-24 09:40:00 | 96.07 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-11-24 09:30:00 | 96.30 | 2025-11-24 15:20:00 | 95.45 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-11-25 09:30:00 | 93.85 | 2025-11-25 09:55:00 | 94.10 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-02 10:30:00 | 93.60 | 2025-12-02 12:55:00 | 93.39 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-02 10:30:00 | 93.60 | 2025-12-02 15:20:00 | 92.93 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2025-12-10 10:50:00 | 90.76 | 2025-12-10 11:30:00 | 90.44 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-12-10 10:50:00 | 90.76 | 2025-12-10 15:00:00 | 90.76 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 10:25:00 | 89.69 | 2025-12-19 11:20:00 | 89.82 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-22 10:40:00 | 90.55 | 2025-12-22 11:05:00 | 90.79 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-12-22 10:40:00 | 90.55 | 2025-12-22 11:15:00 | 90.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-29 09:45:00 | 93.37 | 2025-12-29 09:50:00 | 93.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-30 11:15:00 | 93.98 | 2025-12-30 11:50:00 | 93.66 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-30 11:15:00 | 93.98 | 2025-12-30 12:00:00 | 93.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:50:00 | 94.16 | 2025-12-31 11:20:00 | 93.92 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-01 10:50:00 | 95.55 | 2026-01-01 11:00:00 | 95.32 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-02 11:05:00 | 96.19 | 2026-01-02 12:30:00 | 96.58 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-01-02 11:05:00 | 96.19 | 2026-01-02 15:20:00 | 97.04 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-01-07 11:15:00 | 94.27 | 2026-01-07 11:30:00 | 94.08 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-14 11:15:00 | 91.41 | 2026-01-14 11:30:00 | 91.68 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-16 10:05:00 | 89.23 | 2026-02-16 13:25:00 | 89.59 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-16 10:05:00 | 89.23 | 2026-02-16 15:20:00 | 89.51 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 09:40:00 | 89.48 | 2026-02-18 10:00:00 | 89.20 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 09:40:00 | 89.48 | 2026-02-18 10:20:00 | 89.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 89.25 | 2026-02-19 12:40:00 | 88.98 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-19 11:15:00 | 89.25 | 2026-02-19 13:35:00 | 89.19 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2026-02-24 09:30:00 | 89.34 | 2026-02-24 10:10:00 | 89.57 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-25 10:50:00 | 90.56 | 2026-02-25 11:05:00 | 90.37 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 11:10:00 | 89.61 | 2026-02-27 12:05:00 | 89.38 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-27 11:10:00 | 89.61 | 2026-02-27 12:45:00 | 89.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 10:25:00 | 88.09 | 2026-03-11 10:35:00 | 87.79 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-12 09:30:00 | 86.90 | 2026-03-12 09:40:00 | 87.49 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-12 09:30:00 | 86.90 | 2026-03-12 14:40:00 | 97.71 | TARGET_HIT | 0.50 | 12.44% |
| SELL | retest1 | 2026-05-05 10:35:00 | 109.48 | 2026-05-05 15:05:00 | 108.70 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-05-05 10:35:00 | 109.48 | 2026-05-05 15:15:00 | 109.48 | STOP_HIT | 0.50 | 0.00% |
