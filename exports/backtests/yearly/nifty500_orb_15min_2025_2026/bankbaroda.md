# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-05 15:25:00 (13575 bars)
- **Last close:** 263.20
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
| ENTRY1 | 76 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 15 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 61
- **Target hits / Stop hits / Partials:** 15 / 61 / 28
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 11.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 22 | 38.6% | 8 | 35 | 14 | 0.13% | 7.4% |
| BUY @ 2nd Alert (retest1) | 57 | 22 | 38.6% | 8 | 35 | 14 | 0.13% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 21 | 44.7% | 7 | 26 | 14 | 0.08% | 3.8% |
| SELL @ 2nd Alert (retest1) | 47 | 21 | 44.7% | 7 | 26 | 14 | 0.08% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 43 | 41.3% | 15 | 61 | 28 | 0.11% | 11.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:00:00 | 237.39 | 236.22 | 0.00 | ORB-long ORB[234.82,236.90] vol=2.0x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-05-16 10:10:00 | 236.78 | 236.44 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:35:00 | 239.16 | 237.85 | 0.00 | ORB-long ORB[236.36,237.86] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-05-19 10:45:00 | 238.51 | 237.90 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:00:00 | 240.00 | 237.89 | 0.00 | ORB-long ORB[235.13,237.65] vol=2.0x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-05-21 11:25:00 | 239.36 | 238.16 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 11:00:00 | 243.02 | 241.93 | 0.00 | ORB-long ORB[240.78,242.60] vol=3.8x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 12:45:00 | 243.87 | 242.34 | 0.00 | T1 1.5R @ 243.87 |
| Stop hit — per-position SL triggered | 2025-05-23 13:35:00 | 243.02 | 242.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 11:05:00 | 238.40 | 239.33 | 0.00 | ORB-short ORB[239.25,242.51] vol=2.3x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 238.87 | 239.29 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:55:00 | 242.14 | 242.53 | 0.00 | ORB-short ORB[242.50,245.00] vol=1.9x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 12:35:00 | 241.46 | 242.29 | 0.00 | T1 1.5R @ 241.46 |
| Stop hit — per-position SL triggered | 2025-05-29 13:00:00 | 242.14 | 242.24 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:25:00 | 253.17 | 253.40 | 0.00 | ORB-short ORB[253.61,256.14] vol=1.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-06-04 10:45:00 | 253.77 | 253.42 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 242.01 | 243.79 | 0.00 | ORB-short ORB[242.11,245.58] vol=1.5x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-06-06 10:25:00 | 243.10 | 243.72 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:35:00 | 248.65 | 247.63 | 0.00 | ORB-long ORB[246.62,248.50] vol=1.6x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:40:00 | 249.73 | 247.99 | 0.00 | T1 1.5R @ 249.73 |
| Stop hit — per-position SL triggered | 2025-06-09 09:50:00 | 248.65 | 248.13 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:10:00 | 240.13 | 240.68 | 0.00 | ORB-short ORB[240.65,241.90] vol=2.9x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 12:00:00 | 239.46 | 240.46 | 0.00 | T1 1.5R @ 239.46 |
| Stop hit — per-position SL triggered | 2025-07-10 12:35:00 | 240.13 | 240.33 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:55:00 | 237.98 | 238.83 | 0.00 | ORB-short ORB[238.00,239.89] vol=2.1x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:25:00 | 237.24 | 238.42 | 0.00 | T1 1.5R @ 237.24 |
| Stop hit — per-position SL triggered | 2025-07-11 12:30:00 | 237.98 | 237.77 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:35:00 | 239.62 | 238.71 | 0.00 | ORB-long ORB[236.44,239.35] vol=2.0x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:40:00 | 240.62 | 239.10 | 0.00 | T1 1.5R @ 240.62 |
| Target hit | 2025-07-14 11:30:00 | 240.35 | 240.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2025-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:00:00 | 244.14 | 242.71 | 0.00 | ORB-long ORB[240.74,242.50] vol=5.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-07-15 10:20:00 | 243.54 | 243.30 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:35:00 | 246.24 | 245.31 | 0.00 | ORB-long ORB[243.10,245.95] vol=2.2x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-07-16 09:45:00 | 245.69 | 245.46 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:10:00 | 247.25 | 248.39 | 0.00 | ORB-short ORB[248.52,250.57] vol=2.0x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 11:55:00 | 246.47 | 248.22 | 0.00 | T1 1.5R @ 246.47 |
| Target hit | 2025-07-17 15:20:00 | 246.39 | 247.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:40:00 | 241.71 | 242.89 | 0.00 | ORB-short ORB[242.60,244.22] vol=1.8x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 242.24 | 242.46 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 11:10:00 | 241.18 | 239.83 | 0.00 | ORB-long ORB[239.10,241.10] vol=3.4x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 240.65 | 239.89 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:10:00 | 240.93 | 242.34 | 0.00 | ORB-short ORB[242.55,243.90] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:20:00 | 240.16 | 241.25 | 0.00 | T1 1.5R @ 240.16 |
| Stop hit — per-position SL triggered | 2025-07-24 13:20:00 | 240.93 | 240.82 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 11:15:00 | 242.80 | 241.61 | 0.00 | ORB-long ORB[239.84,242.34] vol=2.4x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-07-29 13:40:00 | 242.09 | 241.94 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:00:00 | 241.47 | 242.18 | 0.00 | ORB-short ORB[242.04,243.90] vol=2.5x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 242.08 | 241.81 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 10:00:00 | 236.64 | 238.56 | 0.00 | ORB-short ORB[237.50,240.20] vol=2.8x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-07-31 10:15:00 | 237.27 | 238.25 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 10:55:00 | 238.15 | 237.06 | 0.00 | ORB-long ORB[236.29,237.95] vol=1.6x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-08-01 12:00:00 | 237.55 | 237.35 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:50:00 | 240.07 | 241.37 | 0.00 | ORB-short ORB[240.64,243.30] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 240.68 | 241.15 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:55:00 | 242.08 | 241.68 | 0.00 | ORB-long ORB[240.41,242.00] vol=3.3x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-08-07 10:05:00 | 241.55 | 241.65 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:45:00 | 242.91 | 243.86 | 0.00 | ORB-short ORB[243.00,244.88] vol=2.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:40:00 | 242.33 | 243.46 | 0.00 | T1 1.5R @ 242.33 |
| Target hit | 2025-08-13 15:20:00 | 241.72 | 242.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-08-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:20:00 | 243.09 | 242.41 | 0.00 | ORB-long ORB[241.40,242.74] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-08-14 11:45:00 | 242.61 | 242.52 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:15:00 | 244.22 | 242.80 | 0.00 | ORB-long ORB[242.13,243.40] vol=1.5x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-08-19 10:40:00 | 243.76 | 243.00 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:15:00 | 248.38 | 247.52 | 0.00 | ORB-long ORB[246.80,248.14] vol=1.9x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-08-20 10:20:00 | 247.88 | 247.56 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:30:00 | 246.92 | 245.90 | 0.00 | ORB-long ORB[244.90,245.86] vol=1.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-08-21 11:30:00 | 246.56 | 246.25 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:50:00 | 242.45 | 243.29 | 0.00 | ORB-short ORB[243.01,244.57] vol=2.0x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:15:00 | 241.80 | 242.96 | 0.00 | T1 1.5R @ 241.80 |
| Target hit | 2025-08-22 15:20:00 | 240.28 | 241.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 237.67 | 239.15 | 0.00 | ORB-short ORB[238.80,241.08] vol=1.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-08-26 09:40:00 | 238.15 | 239.01 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 09:30:00 | 238.25 | 239.04 | 0.00 | ORB-short ORB[238.44,240.78] vol=1.7x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 10:15:00 | 237.50 | 238.66 | 0.00 | T1 1.5R @ 237.50 |
| Target hit | 2025-09-04 15:20:00 | 234.16 | 236.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 11:00:00 | 234.08 | 234.26 | 0.00 | ORB-short ORB[234.26,235.56] vol=2.0x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 11:15:00 | 233.40 | 234.15 | 0.00 | T1 1.5R @ 233.40 |
| Target hit | 2025-09-05 13:30:00 | 234.00 | 233.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2025-09-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:10:00 | 237.21 | 236.27 | 0.00 | ORB-long ORB[235.35,236.64] vol=1.5x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 10:25:00 | 237.92 | 236.71 | 0.00 | T1 1.5R @ 237.92 |
| Target hit | 2025-09-10 11:10:00 | 237.92 | 237.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2025-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:35:00 | 241.35 | 239.70 | 0.00 | ORB-long ORB[237.65,240.25] vol=1.9x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-09-11 09:40:00 | 240.70 | 239.84 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:55:00 | 237.55 | 238.11 | 0.00 | ORB-short ORB[237.60,239.26] vol=8.6x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:05:00 | 236.93 | 238.03 | 0.00 | T1 1.5R @ 236.93 |
| Target hit | 2025-09-12 13:25:00 | 237.31 | 236.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — SELL (started 2025-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:40:00 | 240.40 | 240.80 | 0.00 | ORB-short ORB[240.50,241.50] vol=2.0x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-09-17 10:20:00 | 240.80 | 240.64 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:50:00 | 250.68 | 249.12 | 0.00 | ORB-long ORB[246.01,249.40] vol=1.9x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:00:00 | 252.06 | 249.68 | 0.00 | T1 1.5R @ 252.06 |
| Stop hit — per-position SL triggered | 2025-09-18 10:50:00 | 250.68 | 250.26 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:50:00 | 252.40 | 251.24 | 0.00 | ORB-long ORB[249.00,251.90] vol=3.3x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:20:00 | 253.49 | 251.86 | 0.00 | T1 1.5R @ 253.49 |
| Stop hit — per-position SL triggered | 2025-09-19 10:40:00 | 252.40 | 252.21 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:15:00 | 258.34 | 255.89 | 0.00 | ORB-long ORB[253.80,257.22] vol=1.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-09-24 10:25:00 | 257.39 | 256.05 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:00:00 | 250.23 | 250.89 | 0.00 | ORB-short ORB[250.27,252.98] vol=2.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-09-26 11:05:00 | 250.89 | 250.87 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:40:00 | 251.87 | 251.28 | 0.00 | ORB-long ORB[248.31,250.85] vol=1.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-09-29 10:50:00 | 251.33 | 251.38 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:15:00 | 256.80 | 258.98 | 0.00 | ORB-short ORB[258.45,261.05] vol=1.6x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-10-01 11:35:00 | 257.56 | 258.77 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:50:00 | 259.50 | 260.86 | 0.00 | ORB-short ORB[260.00,262.65] vol=1.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-10-03 11:40:00 | 260.35 | 260.25 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:45:00 | 287.30 | 285.11 | 0.00 | ORB-long ORB[284.00,285.80] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-11-07 10:55:00 | 286.02 | 285.23 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:55:00 | 286.10 | 288.09 | 0.00 | ORB-short ORB[286.30,288.85] vol=1.8x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 11:35:00 | 285.08 | 287.65 | 0.00 | T1 1.5R @ 285.08 |
| Stop hit — per-position SL triggered | 2025-11-12 11:55:00 | 286.10 | 287.44 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:55:00 | 285.70 | 285.20 | 0.00 | ORB-long ORB[281.30,285.20] vol=2.1x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-11-14 11:05:00 | 285.07 | 285.22 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 09:30:00 | 289.60 | 288.00 | 0.00 | ORB-long ORB[285.00,288.95] vol=2.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-11-18 09:35:00 | 288.83 | 288.10 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:10:00 | 290.40 | 288.58 | 0.00 | ORB-long ORB[287.00,288.95] vol=1.6x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-11-19 10:45:00 | 289.68 | 289.38 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:35:00 | 292.75 | 290.76 | 0.00 | ORB-long ORB[286.95,289.20] vol=2.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-11-26 10:40:00 | 292.06 | 290.81 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:50:00 | 290.95 | 288.87 | 0.00 | ORB-long ORB[287.00,289.10] vol=1.8x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:15:00 | 292.22 | 289.98 | 0.00 | T1 1.5R @ 292.22 |
| Stop hit — per-position SL triggered | 2025-11-28 12:15:00 | 290.95 | 290.34 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:40:00 | 295.70 | 293.68 | 0.00 | ORB-long ORB[290.00,293.35] vol=7.5x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 09:45:00 | 297.07 | 294.77 | 0.00 | T1 1.5R @ 297.07 |
| Stop hit — per-position SL triggered | 2025-12-01 09:55:00 | 295.70 | 295.17 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:15:00 | 290.15 | 288.46 | 0.00 | ORB-long ORB[286.65,288.55] vol=2.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-12-05 10:25:00 | 289.27 | 288.60 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:15:00 | 286.95 | 289.32 | 0.00 | ORB-short ORB[290.70,293.30] vol=1.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-12-08 11:30:00 | 287.59 | 288.85 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:00:00 | 287.85 | 289.13 | 0.00 | ORB-short ORB[288.00,290.50] vol=1.8x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:30:00 | 286.98 | 288.85 | 0.00 | T1 1.5R @ 286.98 |
| Target hit | 2025-12-10 15:20:00 | 285.25 | 287.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:35:00 | 284.60 | 286.79 | 0.00 | ORB-short ORB[286.20,288.65] vol=2.3x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-12-11 10:20:00 | 285.58 | 285.80 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 282.10 | 282.76 | 0.00 | ORB-short ORB[282.60,284.85] vol=4.5x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:15:00 | 281.34 | 282.66 | 0.00 | T1 1.5R @ 281.34 |
| Stop hit — per-position SL triggered | 2025-12-16 13:45:00 | 282.10 | 282.01 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 10:15:00 | 285.80 | 285.26 | 0.00 | ORB-long ORB[282.10,285.50] vol=7.9x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-12-17 10:35:00 | 285.02 | 285.41 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 288.50 | 289.75 | 0.00 | ORB-short ORB[288.80,291.55] vol=2.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-12-19 10:55:00 | 289.12 | 289.58 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:15:00 | 295.30 | 294.02 | 0.00 | ORB-long ORB[292.25,293.85] vol=2.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-12-22 10:20:00 | 294.63 | 294.09 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:50:00 | 293.35 | 293.99 | 0.00 | ORB-short ORB[293.75,295.00] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-12-23 11:10:00 | 293.78 | 293.87 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 288.50 | 288.00 | 0.00 | ORB-long ORB[285.60,287.00] vol=3.3x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-12-30 10:55:00 | 287.84 | 287.99 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:50:00 | 298.80 | 297.31 | 0.00 | ORB-long ORB[295.65,297.50] vol=1.7x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:00:00 | 299.96 | 298.99 | 0.00 | T1 1.5R @ 299.96 |
| Target hit | 2026-01-01 10:45:00 | 300.40 | 300.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — SELL (started 2026-02-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:05:00 | 288.55 | 289.28 | 0.00 | ORB-short ORB[289.50,291.30] vol=1.5x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 289.18 | 289.19 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:05:00 | 291.50 | 290.55 | 0.00 | ORB-long ORB[289.25,291.20] vol=2.7x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 290.80 | 290.67 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 288.20 | 286.67 | 0.00 | ORB-long ORB[284.20,286.60] vol=4.3x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:40:00 | 289.11 | 287.21 | 0.00 | T1 1.5R @ 289.11 |
| Target hit | 2026-02-16 15:20:00 | 292.50 | 290.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 295.40 | 294.02 | 0.00 | ORB-long ORB[291.20,295.00] vol=1.8x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 296.72 | 294.36 | 0.00 | T1 1.5R @ 296.72 |
| Target hit | 2026-02-17 15:20:00 | 303.30 | 300.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 306.65 | 306.20 | 0.00 | ORB-long ORB[302.55,306.60] vol=2.4x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:25:00 | 307.97 | 306.59 | 0.00 | T1 1.5R @ 307.97 |
| Target hit | 2026-02-20 14:10:00 | 307.60 | 307.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 69 — BUY (started 2026-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:00:00 | 289.55 | 286.57 | 0.00 | ORB-long ORB[282.95,287.00] vol=3.0x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-03-12 11:05:00 | 288.53 | 286.66 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 282.50 | 284.85 | 0.00 | ORB-short ORB[285.50,287.50] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 283.41 | 284.77 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 272.27 | 270.25 | 0.00 | ORB-long ORB[268.01,271.88] vol=2.3x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:30:00 | 274.72 | 271.17 | 0.00 | T1 1.5R @ 274.72 |
| Target hit | 2026-04-08 15:20:00 | 275.79 | 273.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2026-04-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:40:00 | 271.32 | 269.29 | 0.00 | ORB-long ORB[266.70,270.48] vol=1.7x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 12:00:00 | 272.81 | 270.25 | 0.00 | T1 1.5R @ 272.81 |
| Target hit | 2026-04-13 15:20:00 | 275.60 | 272.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2026-04-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:10:00 | 278.06 | 279.83 | 0.00 | ORB-short ORB[279.02,283.00] vol=6.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-04-15 11:20:00 | 278.93 | 279.77 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 284.46 | 283.83 | 0.00 | ORB-long ORB[281.36,284.40] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 283.83 | 283.86 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:15:00 | 262.31 | 263.66 | 0.00 | ORB-short ORB[263.30,266.22] vol=2.0x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:45:00 | 261.08 | 262.48 | 0.00 | T1 1.5R @ 261.08 |
| Stop hit — per-position SL triggered | 2026-04-30 12:25:00 | 262.31 | 262.27 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 270.70 | 268.89 | 0.00 | ORB-long ORB[264.90,268.40] vol=1.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 269.64 | 269.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 10:00:00 | 237.39 | 2025-05-16 10:10:00 | 236.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-19 10:35:00 | 239.16 | 2025-05-19 10:45:00 | 238.51 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-21 11:00:00 | 240.00 | 2025-05-21 11:25:00 | 239.36 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-23 11:00:00 | 243.02 | 2025-05-23 12:45:00 | 243.87 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-05-23 11:00:00 | 243.02 | 2025-05-23 13:35:00 | 243.02 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 11:05:00 | 238.40 | 2025-05-27 11:15:00 | 238.87 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-05-29 10:55:00 | 242.14 | 2025-05-29 12:35:00 | 241.46 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-05-29 10:55:00 | 242.14 | 2025-05-29 13:00:00 | 242.14 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 10:25:00 | 253.17 | 2025-06-04 10:45:00 | 253.77 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-06 10:15:00 | 242.01 | 2025-06-06 10:25:00 | 243.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-06-09 09:35:00 | 248.65 | 2025-06-09 09:40:00 | 249.73 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-09 09:35:00 | 248.65 | 2025-06-09 09:50:00 | 248.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 11:10:00 | 240.13 | 2025-07-10 12:00:00 | 239.46 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-10 11:10:00 | 240.13 | 2025-07-10 12:35:00 | 240.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 09:55:00 | 237.98 | 2025-07-11 10:25:00 | 237.24 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-11 09:55:00 | 237.98 | 2025-07-11 12:30:00 | 237.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 09:35:00 | 239.62 | 2025-07-14 09:40:00 | 240.62 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-14 09:35:00 | 239.62 | 2025-07-14 11:30:00 | 240.35 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-15 10:00:00 | 244.14 | 2025-07-15 10:20:00 | 243.54 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-16 09:35:00 | 246.24 | 2025-07-16 09:45:00 | 245.69 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-17 11:10:00 | 247.25 | 2025-07-17 11:55:00 | 246.47 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-17 11:10:00 | 247.25 | 2025-07-17 15:20:00 | 246.39 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-22 09:40:00 | 241.71 | 2025-07-22 09:45:00 | 242.24 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-23 11:10:00 | 241.18 | 2025-07-23 11:15:00 | 240.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-24 10:10:00 | 240.93 | 2025-07-24 11:20:00 | 240.16 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-24 10:10:00 | 240.93 | 2025-07-24 13:20:00 | 240.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-29 11:15:00 | 242.80 | 2025-07-29 13:40:00 | 242.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-30 10:00:00 | 241.47 | 2025-07-30 11:15:00 | 242.08 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-31 10:00:00 | 236.64 | 2025-07-31 10:15:00 | 237.27 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-01 10:55:00 | 238.15 | 2025-08-01 12:00:00 | 237.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-06 10:50:00 | 240.07 | 2025-08-06 12:15:00 | 240.68 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-07 09:55:00 | 242.08 | 2025-08-07 10:05:00 | 241.55 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-13 10:45:00 | 242.91 | 2025-08-13 11:40:00 | 242.33 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-08-13 10:45:00 | 242.91 | 2025-08-13 15:20:00 | 241.72 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2025-08-14 10:20:00 | 243.09 | 2025-08-14 11:45:00 | 242.61 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-08-19 10:15:00 | 244.22 | 2025-08-19 10:40:00 | 243.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-20 10:15:00 | 248.38 | 2025-08-20 10:20:00 | 247.88 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-08-21 10:30:00 | 246.92 | 2025-08-21 11:30:00 | 246.56 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-08-22 10:50:00 | 242.45 | 2025-08-22 11:15:00 | 241.80 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-08-22 10:50:00 | 242.45 | 2025-08-22 15:20:00 | 240.28 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2025-08-26 09:35:00 | 237.67 | 2025-08-26 09:40:00 | 238.15 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-04 09:30:00 | 238.25 | 2025-09-04 10:15:00 | 237.50 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-09-04 09:30:00 | 238.25 | 2025-09-04 15:20:00 | 234.16 | TARGET_HIT | 0.50 | 1.72% |
| SELL | retest1 | 2025-09-05 11:00:00 | 234.08 | 2025-09-05 11:15:00 | 233.40 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-05 11:00:00 | 234.08 | 2025-09-05 13:30:00 | 234.00 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2025-09-10 10:10:00 | 237.21 | 2025-09-10 10:25:00 | 237.92 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-10 10:10:00 | 237.21 | 2025-09-10 11:10:00 | 237.92 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-11 09:35:00 | 241.35 | 2025-09-11 09:40:00 | 240.70 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-12 10:55:00 | 237.55 | 2025-09-12 11:05:00 | 236.93 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-09-12 10:55:00 | 237.55 | 2025-09-12 13:25:00 | 237.31 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-09-17 09:40:00 | 240.40 | 2025-09-17 10:20:00 | 240.80 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-18 09:50:00 | 250.68 | 2025-09-18 10:00:00 | 252.06 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-09-18 09:50:00 | 250.68 | 2025-09-18 10:50:00 | 250.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 09:50:00 | 252.40 | 2025-09-19 10:20:00 | 253.49 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-19 09:50:00 | 252.40 | 2025-09-19 10:40:00 | 252.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-24 10:15:00 | 258.34 | 2025-09-24 10:25:00 | 257.39 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-26 11:00:00 | 250.23 | 2025-09-26 11:05:00 | 250.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-29 10:40:00 | 251.87 | 2025-09-29 10:50:00 | 251.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-01 11:15:00 | 256.80 | 2025-10-01 11:35:00 | 257.56 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-03 10:50:00 | 259.50 | 2025-10-03 11:40:00 | 260.35 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-07 10:45:00 | 287.30 | 2025-11-07 10:55:00 | 286.02 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-11-12 10:55:00 | 286.10 | 2025-11-12 11:35:00 | 285.08 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-12 10:55:00 | 286.10 | 2025-11-12 11:55:00 | 286.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 10:55:00 | 285.70 | 2025-11-14 11:05:00 | 285.07 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-18 09:30:00 | 289.60 | 2025-11-18 09:35:00 | 288.83 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-19 10:10:00 | 290.40 | 2025-11-19 10:45:00 | 289.68 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-26 10:35:00 | 292.75 | 2025-11-26 10:40:00 | 292.06 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-28 10:50:00 | 290.95 | 2025-11-28 11:15:00 | 292.22 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-28 10:50:00 | 290.95 | 2025-11-28 12:15:00 | 290.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 09:40:00 | 295.70 | 2025-12-01 09:45:00 | 297.07 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-01 09:40:00 | 295.70 | 2025-12-01 09:55:00 | 295.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:15:00 | 290.15 | 2025-12-05 10:25:00 | 289.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-08 11:15:00 | 286.95 | 2025-12-08 11:30:00 | 287.59 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-10 11:00:00 | 287.85 | 2025-12-10 11:30:00 | 286.98 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-10 11:00:00 | 287.85 | 2025-12-10 15:20:00 | 285.25 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2025-12-11 09:35:00 | 284.60 | 2025-12-11 10:20:00 | 285.58 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-16 11:00:00 | 282.10 | 2025-12-16 11:15:00 | 281.34 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-16 11:00:00 | 282.10 | 2025-12-16 13:45:00 | 282.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 10:15:00 | 285.80 | 2025-12-17 10:35:00 | 285.02 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-19 10:45:00 | 288.50 | 2025-12-19 10:55:00 | 289.12 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-22 10:15:00 | 295.30 | 2025-12-22 10:20:00 | 294.63 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-23 10:50:00 | 293.35 | 2025-12-23 11:10:00 | 293.78 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-30 10:50:00 | 288.50 | 2025-12-30 10:55:00 | 287.84 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-01 09:50:00 | 298.80 | 2026-01-01 10:00:00 | 299.96 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-01 09:50:00 | 298.80 | 2026-01-01 10:45:00 | 300.40 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-06 10:05:00 | 288.55 | 2026-02-06 10:15:00 | 289.18 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-12 10:05:00 | 291.50 | 2026-02-12 10:15:00 | 290.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-16 11:00:00 | 288.20 | 2026-02-16 11:40:00 | 289.11 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-16 11:00:00 | 288.20 | 2026-02-16 15:20:00 | 292.50 | TARGET_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2026-02-17 10:20:00 | 295.40 | 2026-02-17 10:30:00 | 296.72 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-17 10:20:00 | 295.40 | 2026-02-17 15:20:00 | 303.30 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2026-02-20 10:40:00 | 306.65 | 2026-02-20 11:25:00 | 307.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-20 10:40:00 | 306.65 | 2026-02-20 14:10:00 | 307.60 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2026-03-12 11:00:00 | 289.55 | 2026-03-12 11:05:00 | 288.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 10:45:00 | 282.50 | 2026-03-13 10:50:00 | 283.41 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-08 09:35:00 | 272.27 | 2026-04-08 10:30:00 | 274.72 | PARTIAL | 0.50 | 0.90% |
| BUY | retest1 | 2026-04-08 09:35:00 | 272.27 | 2026-04-08 15:20:00 | 275.79 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2026-04-13 10:40:00 | 271.32 | 2026-04-13 12:00:00 | 272.81 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-13 10:40:00 | 271.32 | 2026-04-13 15:20:00 | 275.60 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2026-04-15 11:10:00 | 278.06 | 2026-04-15 11:20:00 | 278.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-22 10:55:00 | 284.46 | 2026-04-22 11:05:00 | 283.83 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-30 10:15:00 | 262.31 | 2026-04-30 11:45:00 | 261.08 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-30 10:15:00 | 262.31 | 2026-04-30 12:25:00 | 262.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:40:00 | 270.70 | 2026-05-04 09:50:00 | 269.64 | STOP_HIT | 1.00 | -0.39% |
