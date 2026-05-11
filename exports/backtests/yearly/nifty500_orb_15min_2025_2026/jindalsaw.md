# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-05 15:25:00 (13650 bars)
- **Last close:** 233.40
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
| ENTRY1 | 45 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 6 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 39
- **Target hits / Stop hits / Partials:** 6 / 39 / 16
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 4.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 7 | 29.2% | 2 | 17 | 5 | -0.05% | -1.2% |
| BUY @ 2nd Alert (retest1) | 24 | 7 | 29.2% | 2 | 17 | 5 | -0.05% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 15 | 40.5% | 4 | 22 | 11 | 0.16% | 5.9% |
| SELL @ 2nd Alert (retest1) | 37 | 15 | 40.5% | 4 | 22 | 11 | 0.16% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 61 | 22 | 36.1% | 6 | 39 | 16 | 0.08% | 4.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:25:00 | 215.50 | 217.41 | 0.00 | ORB-short ORB[216.50,219.30] vol=2.1x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-05-26 14:35:00 | 216.33 | 216.46 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 11:15:00 | 216.63 | 215.48 | 0.00 | ORB-long ORB[213.50,216.00] vol=2.5x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-05-28 11:20:00 | 216.02 | 215.50 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:00:00 | 214.98 | 213.27 | 0.00 | ORB-long ORB[210.98,212.94] vol=2.1x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-06-02 10:20:00 | 214.14 | 213.43 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 11:15:00 | 218.62 | 216.30 | 0.00 | ORB-long ORB[215.00,217.79] vol=2.9x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-06-05 11:20:00 | 217.88 | 216.65 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:45:00 | 219.55 | 217.73 | 0.00 | ORB-long ORB[216.00,218.83] vol=2.3x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 09:50:00 | 220.90 | 218.40 | 0.00 | T1 1.5R @ 220.90 |
| Stop hit — per-position SL triggered | 2025-06-06 09:55:00 | 219.55 | 218.44 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:30:00 | 239.48 | 238.33 | 0.00 | ORB-long ORB[236.10,239.45] vol=1.6x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-06-25 11:20:00 | 238.42 | 239.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 10:05:00 | 240.77 | 242.27 | 0.00 | ORB-short ORB[241.51,244.00] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:35:00 | 239.55 | 241.33 | 0.00 | T1 1.5R @ 239.55 |
| Stop hit — per-position SL triggered | 2025-06-27 11:15:00 | 240.77 | 241.05 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:30:00 | 244.54 | 242.90 | 0.00 | ORB-long ORB[240.48,243.81] vol=1.7x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-07-01 09:35:00 | 243.62 | 243.11 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 234.49 | 236.43 | 0.00 | ORB-short ORB[238.72,241.00] vol=9.2x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-07-02 11:50:00 | 235.64 | 235.31 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 11:15:00 | 231.60 | 232.67 | 0.00 | ORB-short ORB[231.63,234.01] vol=2.3x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:45:00 | 230.87 | 232.43 | 0.00 | T1 1.5R @ 230.87 |
| Stop hit — per-position SL triggered | 2025-07-07 11:55:00 | 231.60 | 232.35 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:05:00 | 230.24 | 231.69 | 0.00 | ORB-short ORB[230.85,233.24] vol=1.8x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-07-08 10:10:00 | 230.92 | 231.65 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:55:00 | 226.39 | 227.81 | 0.00 | ORB-short ORB[228.17,229.50] vol=1.6x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:10:00 | 225.59 | 226.96 | 0.00 | T1 1.5R @ 225.59 |
| Target hit | 2025-07-11 15:20:00 | 224.52 | 225.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 224.10 | 225.29 | 0.00 | ORB-short ORB[224.70,226.80] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-07-16 09:45:00 | 224.72 | 224.85 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:15:00 | 229.97 | 229.16 | 0.00 | ORB-long ORB[227.50,229.69] vol=4.5x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-07-17 11:50:00 | 229.56 | 229.29 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:20:00 | 225.76 | 228.19 | 0.00 | ORB-short ORB[228.20,230.00] vol=1.6x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-07-18 10:30:00 | 226.29 | 228.01 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 227.19 | 225.73 | 0.00 | ORB-long ORB[224.50,225.90] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-07-23 09:45:00 | 226.58 | 226.30 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:00:00 | 228.00 | 227.19 | 0.00 | ORB-long ORB[226.52,227.90] vol=1.6x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:05:00 | 228.79 | 227.74 | 0.00 | T1 1.5R @ 228.79 |
| Target hit | 2025-07-24 11:20:00 | 228.51 | 228.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2025-07-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:25:00 | 225.64 | 226.30 | 0.00 | ORB-short ORB[226.50,227.88] vol=2.5x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:35:00 | 224.85 | 226.21 | 0.00 | T1 1.5R @ 224.85 |
| Stop hit — per-position SL triggered | 2025-07-25 10:40:00 | 225.64 | 226.20 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:55:00 | 218.23 | 216.58 | 0.00 | ORB-long ORB[213.71,216.83] vol=3.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-07-29 11:00:00 | 217.54 | 216.65 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 11:00:00 | 208.51 | 210.11 | 0.00 | ORB-short ORB[210.00,211.90] vol=6.4x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:45:00 | 207.50 | 209.43 | 0.00 | T1 1.5R @ 207.50 |
| Target hit | 2025-08-01 15:20:00 | 206.01 | 207.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2025-08-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:20:00 | 202.00 | 203.07 | 0.00 | ORB-short ORB[202.50,204.78] vol=2.3x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:35:00 | 201.10 | 202.78 | 0.00 | T1 1.5R @ 201.10 |
| Target hit | 2025-08-12 15:20:00 | 198.75 | 200.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2025-08-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:55:00 | 199.35 | 201.01 | 0.00 | ORB-short ORB[201.05,202.64] vol=2.0x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 10:00:00 | 198.42 | 200.77 | 0.00 | T1 1.5R @ 198.42 |
| Stop hit — per-position SL triggered | 2025-08-14 10:05:00 | 199.35 | 200.70 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:15:00 | 201.98 | 200.37 | 0.00 | ORB-long ORB[199.94,201.89] vol=2.6x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-08-18 10:30:00 | 201.27 | 200.67 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 201.72 | 202.82 | 0.00 | ORB-short ORB[201.90,204.50] vol=2.0x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-08-19 10:00:00 | 202.37 | 202.74 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:35:00 | 207.49 | 206.45 | 0.00 | ORB-long ORB[205.15,207.00] vol=3.3x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 10:00:00 | 208.58 | 207.20 | 0.00 | T1 1.5R @ 208.58 |
| Stop hit — per-position SL triggered | 2025-08-21 10:25:00 | 207.49 | 207.31 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:55:00 | 203.13 | 203.97 | 0.00 | ORB-short ORB[204.27,206.90] vol=2.7x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-08-22 12:35:00 | 203.62 | 203.76 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:50:00 | 205.66 | 204.56 | 0.00 | ORB-long ORB[203.55,205.21] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-08-25 10:00:00 | 205.01 | 204.60 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 200.80 | 202.78 | 0.00 | ORB-short ORB[202.20,205.09] vol=2.3x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 201.37 | 202.26 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:40:00 | 198.41 | 199.43 | 0.00 | ORB-short ORB[199.10,201.51] vol=2.2x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:50:00 | 197.39 | 198.99 | 0.00 | T1 1.5R @ 197.39 |
| Stop hit — per-position SL triggered | 2025-08-29 09:55:00 | 198.41 | 198.87 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 205.21 | 203.40 | 0.00 | ORB-long ORB[202.40,203.85] vol=2.2x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 204.48 | 203.65 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:50:00 | 207.78 | 204.56 | 0.00 | ORB-long ORB[202.75,205.55] vol=7.3x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:55:00 | 208.94 | 205.92 | 0.00 | T1 1.5R @ 208.94 |
| Stop hit — per-position SL triggered | 2025-09-12 10:05:00 | 207.78 | 206.45 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:35:00 | 202.25 | 203.42 | 0.00 | ORB-short ORB[202.30,204.70] vol=2.4x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-09-30 10:55:00 | 202.82 | 203.34 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 201.41 | 202.58 | 0.00 | ORB-short ORB[202.31,205.10] vol=2.0x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:35:00 | 200.87 | 202.32 | 0.00 | T1 1.5R @ 200.87 |
| Stop hit — per-position SL triggered | 2025-10-07 12:05:00 | 201.41 | 202.23 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:15:00 | 169.49 | 170.35 | 0.00 | ORB-short ORB[169.62,171.82] vol=2.9x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-11-11 11:50:00 | 169.89 | 170.32 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:55:00 | 172.18 | 171.80 | 0.00 | ORB-long ORB[171.05,172.05] vol=2.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-11-13 11:20:00 | 171.84 | 171.84 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:00:00 | 164.85 | 165.63 | 0.00 | ORB-short ORB[165.82,167.45] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-11-20 10:10:00 | 165.18 | 165.56 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:45:00 | 165.24 | 165.66 | 0.00 | ORB-short ORB[165.60,167.25] vol=2.4x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:05:00 | 164.66 | 165.52 | 0.00 | T1 1.5R @ 164.66 |
| Stop hit — per-position SL triggered | 2025-11-21 11:25:00 | 165.24 | 165.20 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 167.14 | 168.22 | 0.00 | ORB-short ORB[167.52,169.55] vol=1.6x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-11-27 10:00:00 | 167.74 | 167.73 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:05:00 | 162.99 | 163.89 | 0.00 | ORB-short ORB[163.21,165.20] vol=2.1x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-12-03 11:35:00 | 163.41 | 163.77 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:00:00 | 165.58 | 164.12 | 0.00 | ORB-long ORB[163.00,165.30] vol=2.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-12-04 11:05:00 | 165.10 | 164.24 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:05:00 | 160.57 | 161.39 | 0.00 | ORB-short ORB[161.30,162.69] vol=1.5x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 159.92 | 161.07 | 0.00 | T1 1.5R @ 159.92 |
| Target hit | 2025-12-08 15:20:00 | 156.86 | 158.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-12-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:50:00 | 157.91 | 158.75 | 0.00 | ORB-short ORB[158.50,160.81] vol=1.8x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 158.24 | 158.69 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:05:00 | 162.80 | 161.38 | 0.00 | ORB-long ORB[160.51,162.14] vol=2.3x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:10:00 | 163.55 | 161.59 | 0.00 | T1 1.5R @ 163.55 |
| Target hit | 2025-12-18 14:00:00 | 163.49 | 163.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2025-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:55:00 | 166.00 | 165.14 | 0.00 | ORB-long ORB[164.32,165.86] vol=9.7x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 165.53 | 165.20 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 221.21 | 222.45 | 0.00 | ORB-short ORB[221.65,224.85] vol=1.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-04-17 09:35:00 | 222.07 | 222.00 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-26 10:25:00 | 215.50 | 2025-05-26 14:35:00 | 216.33 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-05-28 11:15:00 | 216.63 | 2025-05-28 11:20:00 | 216.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-02 10:00:00 | 214.98 | 2025-06-02 10:20:00 | 214.14 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-05 11:15:00 | 218.62 | 2025-06-05 11:20:00 | 217.88 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-06 09:45:00 | 219.55 | 2025-06-06 09:50:00 | 220.90 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-06-06 09:45:00 | 219.55 | 2025-06-06 09:55:00 | 219.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 09:30:00 | 239.48 | 2025-06-25 11:20:00 | 238.42 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-06-27 10:05:00 | 240.77 | 2025-06-27 10:35:00 | 239.55 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-06-27 10:05:00 | 240.77 | 2025-06-27 11:15:00 | 240.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-01 09:30:00 | 244.54 | 2025-07-01 09:35:00 | 243.62 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-07-02 09:40:00 | 234.49 | 2025-07-02 11:50:00 | 235.64 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-07-07 11:15:00 | 231.60 | 2025-07-07 11:45:00 | 230.87 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-07 11:15:00 | 231.60 | 2025-07-07 11:55:00 | 231.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 10:05:00 | 230.24 | 2025-07-08 10:10:00 | 230.92 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-11 10:55:00 | 226.39 | 2025-07-11 11:10:00 | 225.59 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-11 10:55:00 | 226.39 | 2025-07-11 15:20:00 | 224.52 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-07-16 09:30:00 | 224.10 | 2025-07-16 09:45:00 | 224.72 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-17 11:15:00 | 229.97 | 2025-07-17 11:50:00 | 229.56 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-18 10:20:00 | 225.76 | 2025-07-18 10:30:00 | 226.29 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-23 09:30:00 | 227.19 | 2025-07-23 09:45:00 | 226.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-24 10:00:00 | 228.00 | 2025-07-24 10:05:00 | 228.79 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-24 10:00:00 | 228.00 | 2025-07-24 11:20:00 | 228.51 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-25 10:25:00 | 225.64 | 2025-07-25 10:35:00 | 224.85 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-25 10:25:00 | 225.64 | 2025-07-25 10:40:00 | 225.64 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-29 10:55:00 | 218.23 | 2025-07-29 11:00:00 | 217.54 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-01 11:00:00 | 208.51 | 2025-08-01 11:45:00 | 207.50 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-08-01 11:00:00 | 208.51 | 2025-08-01 15:20:00 | 206.01 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2025-08-12 10:20:00 | 202.00 | 2025-08-12 10:35:00 | 201.10 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-12 10:20:00 | 202.00 | 2025-08-12 15:20:00 | 198.75 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2025-08-14 09:55:00 | 199.35 | 2025-08-14 10:00:00 | 198.42 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-08-14 09:55:00 | 199.35 | 2025-08-14 10:05:00 | 199.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-18 10:15:00 | 201.98 | 2025-08-18 10:30:00 | 201.27 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-19 09:50:00 | 201.72 | 2025-08-19 10:00:00 | 202.37 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-21 09:35:00 | 207.49 | 2025-08-21 10:00:00 | 208.58 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-08-21 09:35:00 | 207.49 | 2025-08-21 10:25:00 | 207.49 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 10:55:00 | 203.13 | 2025-08-22 12:35:00 | 203.62 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-25 09:50:00 | 205.66 | 2025-08-25 10:00:00 | 205.01 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-26 09:30:00 | 200.80 | 2025-08-26 09:45:00 | 201.37 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-29 09:40:00 | 198.41 | 2025-08-29 09:50:00 | 197.39 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-08-29 09:40:00 | 198.41 | 2025-08-29 09:55:00 | 198.41 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-10 09:35:00 | 205.21 | 2025-09-10 09:45:00 | 204.48 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-12 09:50:00 | 207.78 | 2025-09-12 09:55:00 | 208.94 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-09-12 09:50:00 | 207.78 | 2025-09-12 10:05:00 | 207.78 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-30 10:35:00 | 202.25 | 2025-09-30 10:55:00 | 202.82 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-07 11:10:00 | 201.41 | 2025-10-07 11:35:00 | 200.87 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-07 11:10:00 | 201.41 | 2025-10-07 12:05:00 | 201.41 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 11:15:00 | 169.49 | 2025-11-11 11:50:00 | 169.89 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-13 10:55:00 | 172.18 | 2025-11-13 11:20:00 | 171.84 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-20 10:00:00 | 164.85 | 2025-11-20 10:10:00 | 165.18 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-21 09:45:00 | 165.24 | 2025-11-21 10:05:00 | 164.66 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-21 09:45:00 | 165.24 | 2025-11-21 11:25:00 | 165.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 09:30:00 | 167.14 | 2025-11-27 10:00:00 | 167.74 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-03 11:05:00 | 162.99 | 2025-12-03 11:35:00 | 163.41 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-04 11:00:00 | 165.58 | 2025-12-04 11:05:00 | 165.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-08 10:05:00 | 160.57 | 2025-12-08 10:15:00 | 159.92 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-08 10:05:00 | 160.57 | 2025-12-08 15:20:00 | 156.86 | TARGET_HIT | 0.50 | 2.31% |
| SELL | retest1 | 2025-12-12 10:50:00 | 157.91 | 2025-12-12 11:15:00 | 158.24 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-18 10:05:00 | 162.80 | 2025-12-18 10:10:00 | 163.55 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-18 10:05:00 | 162.80 | 2025-12-18 14:00:00 | 163.49 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2025-12-30 10:55:00 | 166.00 | 2025-12-30 11:15:00 | 165.53 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-17 09:30:00 | 221.21 | 2026-04-17 09:35:00 | 222.07 | STOP_HIT | 1.00 | -0.39% |
