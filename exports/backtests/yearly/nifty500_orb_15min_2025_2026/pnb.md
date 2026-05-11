# Punjab National Bank (PNB)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 107.20
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
| ENTRY1 | 98 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 16 |
| STOP_HIT | 82 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 82
- **Target hits / Stop hits / Partials:** 16 / 82 / 37
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 18.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 80 | 28 | 35.0% | 8 | 52 | 20 | 0.09% | 7.4% |
| BUY @ 2nd Alert (retest1) | 80 | 28 | 35.0% | 8 | 52 | 20 | 0.09% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 55 | 25 | 45.5% | 8 | 30 | 17 | 0.20% | 11.1% |
| SELL @ 2nd Alert (retest1) | 55 | 25 | 45.5% | 8 | 30 | 17 | 0.20% | 11.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 135 | 53 | 39.3% | 16 | 82 | 37 | 0.14% | 18.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 11:15:00 | 95.64 | 95.03 | 0.00 | ORB-long ORB[94.55,95.51] vol=1.8x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-05-12 12:45:00 | 95.28 | 95.16 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:30:00 | 97.22 | 96.44 | 0.00 | ORB-long ORB[95.40,96.80] vol=3.1x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 09:35:00 | 97.63 | 96.88 | 0.00 | T1 1.5R @ 97.63 |
| Target hit | 2025-05-13 13:15:00 | 97.41 | 97.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 99.73 | 99.37 | 0.00 | ORB-long ORB[99.00,99.48] vol=2.2x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-05-19 09:35:00 | 99.54 | 99.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:15:00 | 101.98 | 101.11 | 0.00 | ORB-long ORB[100.01,100.98] vol=1.6x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-05-21 10:35:00 | 101.66 | 101.23 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 09:35:00 | 99.72 | 100.07 | 0.00 | ORB-short ORB[99.82,100.78] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-05-23 10:00:00 | 100.03 | 100.02 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:35:00 | 102.47 | 102.12 | 0.00 | ORB-long ORB[101.75,102.36] vol=2.0x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-05-28 09:55:00 | 102.24 | 102.24 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:00:00 | 109.59 | 108.87 | 0.00 | ORB-long ORB[108.05,109.39] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-06-03 10:05:00 | 109.28 | 108.89 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 110.33 | 109.60 | 0.00 | ORB-long ORB[108.60,109.84] vol=4.6x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 109.97 | 109.62 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:35:00 | 112.30 | 111.64 | 0.00 | ORB-long ORB[110.76,111.93] vol=2.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:45:00 | 112.86 | 111.90 | 0.00 | T1 1.5R @ 112.86 |
| Stop hit — per-position SL triggered | 2025-06-09 09:50:00 | 112.30 | 111.93 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 105.03 | 105.72 | 0.00 | ORB-short ORB[105.60,106.80] vol=4.1x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-06-16 09:35:00 | 105.32 | 105.65 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:35:00 | 108.32 | 107.47 | 0.00 | ORB-long ORB[106.80,107.91] vol=1.9x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 108.01 | 107.54 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:50:00 | 104.81 | 105.18 | 0.00 | ORB-short ORB[104.85,105.88] vol=2.2x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:05:00 | 104.41 | 104.99 | 0.00 | T1 1.5R @ 104.41 |
| Target hit | 2025-06-19 15:20:00 | 102.91 | 103.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 106.41 | 105.77 | 0.00 | ORB-long ORB[105.08,105.95] vol=3.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-06-24 09:45:00 | 106.07 | 105.91 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:15:00 | 105.88 | 105.47 | 0.00 | ORB-long ORB[105.14,105.75] vol=1.7x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:25:00 | 106.18 | 105.60 | 0.00 | T1 1.5R @ 106.18 |
| Stop hit — per-position SL triggered | 2025-06-25 10:45:00 | 105.88 | 105.66 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 104.95 | 105.76 | 0.00 | ORB-short ORB[105.78,106.54] vol=2.1x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-06-26 11:20:00 | 105.12 | 105.75 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:10:00 | 112.01 | 111.52 | 0.00 | ORB-long ORB[110.63,111.59] vol=2.1x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:20:00 | 112.41 | 111.77 | 0.00 | T1 1.5R @ 112.41 |
| Target hit | 2025-07-07 15:20:00 | 112.38 | 112.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-07-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:10:00 | 110.42 | 110.92 | 0.00 | ORB-short ORB[110.98,111.50] vol=2.7x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:35:00 | 110.14 | 110.83 | 0.00 | T1 1.5R @ 110.14 |
| Target hit | 2025-07-10 15:20:00 | 109.69 | 110.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-07-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:35:00 | 111.75 | 111.14 | 0.00 | ORB-long ORB[110.15,111.25] vol=2.0x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:45:00 | 112.25 | 111.47 | 0.00 | T1 1.5R @ 112.25 |
| Stop hit — per-position SL triggered | 2025-07-14 09:50:00 | 111.75 | 111.49 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:00:00 | 112.64 | 111.72 | 0.00 | ORB-long ORB[110.75,111.49] vol=5.3x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 10:10:00 | 113.07 | 112.11 | 0.00 | T1 1.5R @ 113.07 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 112.64 | 112.17 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 112.82 | 112.66 | 0.00 | ORB-long ORB[111.86,112.78] vol=2.1x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-07-16 09:50:00 | 112.58 | 112.67 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 114.26 | 115.04 | 0.00 | ORB-short ORB[114.65,115.60] vol=1.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-07-17 10:05:00 | 114.62 | 114.82 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 112.07 | 112.55 | 0.00 | ORB-short ORB[112.22,113.10] vol=1.7x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 09:40:00 | 111.70 | 112.38 | 0.00 | T1 1.5R @ 111.70 |
| Target hit | 2025-07-22 15:20:00 | 109.27 | 110.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 11:15:00 | 108.99 | 109.00 | 0.00 | ORB-short ORB[109.21,109.83] vol=2.6x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:30:00 | 108.53 | 108.97 | 0.00 | T1 1.5R @ 108.53 |
| Stop hit — per-position SL triggered | 2025-07-23 12:10:00 | 108.99 | 108.95 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:15:00 | 109.03 | 109.56 | 0.00 | ORB-short ORB[109.83,110.34] vol=1.8x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:20:00 | 108.75 | 109.49 | 0.00 | T1 1.5R @ 108.75 |
| Stop hit — per-position SL triggered | 2025-07-24 13:20:00 | 109.03 | 109.19 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:30:00 | 109.89 | 109.22 | 0.00 | ORB-long ORB[108.94,109.70] vol=2.3x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-07-30 10:35:00 | 109.55 | 109.25 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:55:00 | 104.85 | 106.37 | 0.00 | ORB-short ORB[106.57,107.44] vol=1.6x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 10:15:00 | 104.05 | 105.56 | 0.00 | T1 1.5R @ 104.05 |
| Stop hit — per-position SL triggered | 2025-07-31 11:00:00 | 104.85 | 105.23 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:00:00 | 103.30 | 103.69 | 0.00 | ORB-short ORB[103.50,103.96] vol=2.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 103.56 | 103.58 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:40:00 | 106.41 | 106.71 | 0.00 | ORB-short ORB[106.55,107.23] vol=2.3x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-08-19 09:50:00 | 106.62 | 106.68 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:35:00 | 107.45 | 107.67 | 0.00 | ORB-short ORB[107.50,108.05] vol=1.6x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-08-20 09:45:00 | 107.64 | 107.68 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:30:00 | 107.88 | 107.46 | 0.00 | ORB-long ORB[107.15,107.67] vol=1.6x ATR=0.14 |
| Stop hit — per-position SL triggered | 2025-08-21 10:35:00 | 107.74 | 107.46 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 104.05 | 104.55 | 0.00 | ORB-short ORB[104.30,105.39] vol=1.6x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 104.24 | 104.45 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:45:00 | 100.75 | 101.16 | 0.00 | ORB-short ORB[100.90,101.91] vol=4.7x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-08-29 10:20:00 | 101.07 | 101.06 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:35:00 | 102.81 | 102.39 | 0.00 | ORB-long ORB[101.92,102.80] vol=1.8x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:45:00 | 103.13 | 102.67 | 0.00 | T1 1.5R @ 103.13 |
| Target hit | 2025-09-02 13:15:00 | 103.05 | 103.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 104.61 | 104.15 | 0.00 | ORB-long ORB[103.80,104.41] vol=1.7x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-09-08 09:45:00 | 104.37 | 104.24 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:10:00 | 105.91 | 105.38 | 0.00 | ORB-long ORB[104.60,105.60] vol=1.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-09-10 10:20:00 | 105.69 | 105.44 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:20:00 | 108.87 | 108.44 | 0.00 | ORB-long ORB[108.37,108.84] vol=2.1x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:25:00 | 109.14 | 108.80 | 0.00 | T1 1.5R @ 109.14 |
| Target hit | 2025-09-17 15:20:00 | 111.96 | 110.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-09-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:00:00 | 113.13 | 112.31 | 0.00 | ORB-long ORB[111.40,112.50] vol=1.7x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-09-18 10:05:00 | 112.75 | 112.36 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:50:00 | 113.45 | 112.85 | 0.00 | ORB-long ORB[111.86,113.13] vol=2.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-09-19 10:00:00 | 113.15 | 112.97 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:50:00 | 112.08 | 111.30 | 0.00 | ORB-long ORB[110.53,111.48] vol=1.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-09-25 11:00:00 | 111.78 | 111.33 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:35:00 | 109.39 | 110.08 | 0.00 | ORB-short ORB[109.72,111.34] vol=1.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-09-26 09:45:00 | 109.75 | 110.02 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:55:00 | 109.31 | 109.14 | 0.00 | ORB-long ORB[107.90,108.87] vol=2.1x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:05:00 | 109.66 | 109.16 | 0.00 | T1 1.5R @ 109.66 |
| Stop hit — per-position SL triggered | 2025-09-29 11:25:00 | 109.31 | 109.18 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 09:50:00 | 111.77 | 111.35 | 0.00 | ORB-long ORB[109.85,111.31] vol=3.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-09-30 10:05:00 | 111.44 | 111.40 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:15:00 | 112.20 | 112.78 | 0.00 | ORB-short ORB[112.50,113.56] vol=1.5x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:45:00 | 111.71 | 112.72 | 0.00 | T1 1.5R @ 111.71 |
| Stop hit — per-position SL triggered | 2025-10-01 12:35:00 | 112.20 | 112.43 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 113.84 | 114.62 | 0.00 | ORB-short ORB[114.27,115.60] vol=4.0x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-10-06 11:30:00 | 114.15 | 114.51 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:50:00 | 112.99 | 113.50 | 0.00 | ORB-short ORB[113.12,114.48] vol=2.7x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-10-08 10:00:00 | 113.31 | 113.47 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:15:00 | 114.04 | 113.51 | 0.00 | ORB-long ORB[113.05,113.71] vol=1.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-10-09 10:25:00 | 113.76 | 113.58 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 115.34 | 114.80 | 0.00 | ORB-long ORB[114.16,115.05] vol=2.9x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-10-10 09:35:00 | 115.10 | 114.85 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 116.14 | 116.65 | 0.00 | ORB-short ORB[116.30,117.34] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-10-14 09:55:00 | 116.40 | 116.56 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:05:00 | 116.49 | 116.05 | 0.00 | ORB-long ORB[115.40,116.13] vol=2.0x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:55:00 | 116.83 | 116.20 | 0.00 | T1 1.5R @ 116.83 |
| Stop hit — per-position SL triggered | 2025-10-15 14:20:00 | 116.49 | 116.50 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:40:00 | 115.98 | 115.19 | 0.00 | ORB-long ORB[114.25,115.70] vol=2.0x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:55:00 | 116.54 | 115.44 | 0.00 | T1 1.5R @ 116.54 |
| Target hit | 2025-10-20 15:20:00 | 118.14 | 117.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2025-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:35:00 | 119.59 | 118.76 | 0.00 | ORB-long ORB[118.07,119.05] vol=2.1x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-10-23 09:40:00 | 119.21 | 118.83 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:55:00 | 118.46 | 118.13 | 0.00 | ORB-long ORB[117.57,118.25] vol=2.1x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 118.15 | 118.17 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:30:00 | 122.26 | 121.50 | 0.00 | ORB-long ORB[120.70,121.77] vol=2.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-10-29 10:35:00 | 121.93 | 121.51 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-10-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:55:00 | 120.14 | 120.84 | 0.00 | ORB-short ORB[120.80,121.73] vol=2.1x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-10-30 10:05:00 | 120.48 | 120.80 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 121.58 | 120.83 | 0.00 | ORB-long ORB[120.08,121.20] vol=2.4x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-10-31 09:40:00 | 121.22 | 121.04 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:20:00 | 120.27 | 121.12 | 0.00 | ORB-short ORB[121.48,122.62] vol=2.1x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-11-11 10:25:00 | 120.56 | 121.03 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 123.25 | 122.73 | 0.00 | ORB-long ORB[122.00,122.83] vol=1.7x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-11-12 09:50:00 | 122.99 | 122.74 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:05:00 | 123.12 | 122.87 | 0.00 | ORB-long ORB[122.35,122.89] vol=2.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-11-13 10:10:00 | 122.84 | 122.89 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 122.40 | 121.88 | 0.00 | ORB-long ORB[121.35,122.08] vol=1.8x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-11-14 09:35:00 | 122.13 | 121.93 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 124.09 | 123.50 | 0.00 | ORB-long ORB[122.31,123.86] vol=3.3x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-11-17 09:35:00 | 123.76 | 123.55 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-11-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:55:00 | 124.65 | 125.44 | 0.00 | ORB-short ORB[125.01,126.16] vol=1.7x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-11-20 11:05:00 | 124.91 | 125.38 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-11-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:30:00 | 122.49 | 123.13 | 0.00 | ORB-short ORB[123.08,123.79] vol=1.5x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:45:00 | 122.13 | 122.96 | 0.00 | T1 1.5R @ 122.13 |
| Stop hit — per-position SL triggered | 2025-11-21 11:15:00 | 122.49 | 122.88 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 125.02 | 124.16 | 0.00 | ORB-long ORB[123.35,124.14] vol=2.8x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 10:00:00 | 125.56 | 124.77 | 0.00 | T1 1.5R @ 125.56 |
| Stop hit — per-position SL triggered | 2025-11-26 10:55:00 | 125.02 | 125.09 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:00:00 | 120.58 | 119.80 | 0.00 | ORB-long ORB[119.15,120.09] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-12-05 10:05:00 | 120.26 | 119.99 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:05:00 | 118.72 | 120.18 | 0.00 | ORB-short ORB[121.06,121.68] vol=2.0x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:35:00 | 118.31 | 119.92 | 0.00 | T1 1.5R @ 118.31 |
| Target hit | 2025-12-08 15:20:00 | 115.98 | 117.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 118.68 | 118.39 | 0.00 | ORB-long ORB[117.85,118.62] vol=1.8x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 09:40:00 | 119.09 | 118.57 | 0.00 | T1 1.5R @ 119.09 |
| Stop hit — per-position SL triggered | 2025-12-10 09:50:00 | 118.68 | 118.63 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:35:00 | 116.52 | 117.24 | 0.00 | ORB-short ORB[116.85,117.98] vol=2.4x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-12-11 10:05:00 | 116.90 | 116.98 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 117.68 | 118.16 | 0.00 | ORB-short ORB[118.25,118.89] vol=1.5x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:15:00 | 117.37 | 118.10 | 0.00 | T1 1.5R @ 117.37 |
| Target hit | 2025-12-16 15:20:00 | 117.06 | 117.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2025-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:40:00 | 118.54 | 118.07 | 0.00 | ORB-long ORB[116.92,118.37] vol=1.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-12-17 10:10:00 | 118.26 | 118.29 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:45:00 | 120.83 | 120.62 | 0.00 | ORB-long ORB[120.20,120.64] vol=2.9x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 10:15:00 | 121.18 | 120.76 | 0.00 | T1 1.5R @ 121.18 |
| Stop hit — per-position SL triggered | 2025-12-22 10:30:00 | 120.83 | 120.77 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:40:00 | 120.26 | 120.39 | 0.00 | ORB-short ORB[120.35,120.98] vol=1.7x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:05:00 | 119.96 | 120.32 | 0.00 | T1 1.5R @ 119.96 |
| Target hit | 2025-12-26 12:05:00 | 120.17 | 120.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2026-01-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:20:00 | 124.15 | 123.95 | 0.00 | ORB-long ORB[123.24,124.10] vol=1.8x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-01-01 12:35:00 | 123.82 | 124.02 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 11:15:00 | 126.71 | 126.17 | 0.00 | ORB-long ORB[125.30,126.25] vol=2.5x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-01-07 12:00:00 | 126.47 | 126.26 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 124.07 | 124.88 | 0.00 | ORB-short ORB[124.51,125.80] vol=4.5x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 123.62 | 124.77 | 0.00 | T1 1.5R @ 123.62 |
| Target hit | 2026-01-08 15:20:00 | 122.71 | 123.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-01-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:05:00 | 124.05 | 123.32 | 0.00 | ORB-long ORB[122.40,123.61] vol=1.7x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 10:45:00 | 124.65 | 123.61 | 0.00 | T1 1.5R @ 124.65 |
| Stop hit — per-position SL triggered | 2026-01-09 10:55:00 | 124.05 | 123.64 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:05:00 | 123.10 | 123.36 | 0.00 | ORB-short ORB[123.20,124.09] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-01-13 12:25:00 | 123.39 | 123.21 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:55:00 | 126.18 | 125.29 | 0.00 | ORB-long ORB[124.52,125.71] vol=2.6x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:05:00 | 126.67 | 125.47 | 0.00 | T1 1.5R @ 126.67 |
| Stop hit — per-position SL triggered | 2026-01-14 11:30:00 | 126.18 | 125.63 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-01-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 09:35:00 | 133.98 | 133.02 | 0.00 | ORB-long ORB[131.80,133.33] vol=2.1x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-01-19 09:40:00 | 133.51 | 133.09 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:55:00 | 123.65 | 125.10 | 0.00 | ORB-short ORB[125.25,126.69] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-01-21 11:00:00 | 124.15 | 125.05 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:30:00 | 126.20 | 125.80 | 0.00 | ORB-long ORB[124.58,126.10] vol=3.1x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-01-22 09:50:00 | 125.79 | 125.94 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-01-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 09:45:00 | 126.05 | 125.20 | 0.00 | ORB-long ORB[124.50,125.35] vol=3.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-01-29 09:50:00 | 125.67 | 125.27 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:55:00 | 125.12 | 124.44 | 0.00 | ORB-long ORB[123.26,124.60] vol=1.7x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 124.77 | 124.50 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-02-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:25:00 | 122.62 | 123.61 | 0.00 | ORB-short ORB[123.52,124.43] vol=2.3x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 10:40:00 | 122.09 | 123.29 | 0.00 | T1 1.5R @ 122.09 |
| Stop hit — per-position SL triggered | 2026-02-06 11:40:00 | 122.62 | 122.95 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 121.41 | 122.03 | 0.00 | ORB-short ORB[121.65,122.91] vol=1.8x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 12:00:00 | 121.04 | 121.87 | 0.00 | T1 1.5R @ 121.04 |
| Stop hit — per-position SL triggered | 2026-02-12 12:30:00 | 121.41 | 121.80 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 122.13 | 121.26 | 0.00 | ORB-long ORB[119.81,121.58] vol=1.5x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:20:00 | 122.66 | 121.57 | 0.00 | T1 1.5R @ 122.66 |
| Target hit | 2026-02-17 15:20:00 | 124.91 | 124.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 126.45 | 125.88 | 0.00 | ORB-long ORB[124.83,126.35] vol=2.4x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:40:00 | 127.01 | 126.14 | 0.00 | T1 1.5R @ 127.01 |
| Target hit | 2026-02-18 15:20:00 | 128.15 | 127.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 130.53 | 130.91 | 0.00 | ORB-short ORB[130.56,131.66] vol=1.7x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 130.88 | 130.86 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 117.38 | 117.82 | 0.00 | ORB-short ORB[117.50,118.44] vol=1.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:30:00 | 116.98 | 117.67 | 0.00 | T1 1.5R @ 116.98 |
| Target hit | 2026-03-11 15:20:00 | 115.81 | 117.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2026-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:50:00 | 115.55 | 114.69 | 0.00 | ORB-long ORB[113.66,114.94] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 115.08 | 114.90 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 114.34 | 115.13 | 0.00 | ORB-short ORB[114.55,116.27] vol=1.7x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:55:00 | 113.72 | 114.85 | 0.00 | T1 1.5R @ 113.72 |
| Stop hit — per-position SL triggered | 2026-03-13 11:30:00 | 114.34 | 114.32 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 109.98 | 109.24 | 0.00 | ORB-long ORB[108.55,109.67] vol=1.5x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 14:30:00 | 110.80 | 109.83 | 0.00 | T1 1.5R @ 110.80 |
| Target hit | 2026-04-08 15:20:00 | 111.13 | 110.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 112.86 | 113.46 | 0.00 | ORB-short ORB[113.06,114.05] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 113.27 | 113.44 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-04-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:55:00 | 114.62 | 114.04 | 0.00 | ORB-long ORB[113.05,114.20] vol=2.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 114.37 | 114.13 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 114.32 | 113.87 | 0.00 | ORB-long ORB[112.85,114.10] vol=1.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 114.00 | 113.93 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 114.69 | 114.23 | 0.00 | ORB-long ORB[113.60,114.49] vol=1.6x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 114.43 | 114.31 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 115.08 | 114.49 | 0.00 | ORB-long ORB[113.35,114.55] vol=3.3x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:20:00 | 115.49 | 114.68 | 0.00 | T1 1.5R @ 115.49 |
| Stop hit — per-position SL triggered | 2026-04-22 11:45:00 | 115.08 | 114.74 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 112.68 | 112.13 | 0.00 | ORB-long ORB[111.24,112.60] vol=1.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 112.35 | 112.31 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2026-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:10:00 | 108.85 | 109.48 | 0.00 | ORB-short ORB[109.17,110.80] vol=1.8x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:45:00 | 108.32 | 109.18 | 0.00 | T1 1.5R @ 108.32 |
| Stop hit — per-position SL triggered | 2026-04-30 12:10:00 | 108.85 | 109.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 11:15:00 | 95.64 | 2025-05-12 12:45:00 | 95.28 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-13 09:30:00 | 97.22 | 2025-05-13 09:35:00 | 97.63 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-05-13 09:30:00 | 97.22 | 2025-05-13 13:15:00 | 97.41 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-05-19 09:30:00 | 99.73 | 2025-05-19 09:35:00 | 99.54 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-21 10:15:00 | 101.98 | 2025-05-21 10:35:00 | 101.66 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-05-23 09:35:00 | 99.72 | 2025-05-23 10:00:00 | 100.03 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-05-28 09:35:00 | 102.47 | 2025-05-28 09:55:00 | 102.24 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-03 10:00:00 | 109.59 | 2025-06-03 10:05:00 | 109.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-06 10:05:00 | 110.33 | 2025-06-06 10:10:00 | 109.97 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-09 09:35:00 | 112.30 | 2025-06-09 09:45:00 | 112.86 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-06-09 09:35:00 | 112.30 | 2025-06-09 09:50:00 | 112.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-16 09:30:00 | 105.03 | 2025-06-16 09:35:00 | 105.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-17 09:35:00 | 108.32 | 2025-06-17 09:40:00 | 108.01 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-19 09:50:00 | 104.81 | 2025-06-19 10:05:00 | 104.41 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-19 09:50:00 | 104.81 | 2025-06-19 15:20:00 | 102.91 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2025-06-24 09:30:00 | 106.41 | 2025-06-24 09:45:00 | 106.07 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-25 10:15:00 | 105.88 | 2025-06-25 10:25:00 | 106.18 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-06-25 10:15:00 | 105.88 | 2025-06-25 10:45:00 | 105.88 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 11:15:00 | 104.95 | 2025-06-26 11:20:00 | 105.12 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-07 10:10:00 | 112.01 | 2025-07-07 10:20:00 | 112.41 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-07 10:10:00 | 112.01 | 2025-07-07 15:20:00 | 112.38 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-10 11:10:00 | 110.42 | 2025-07-10 11:35:00 | 110.14 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-10 11:10:00 | 110.42 | 2025-07-10 15:20:00 | 109.69 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-07-14 09:35:00 | 111.75 | 2025-07-14 09:45:00 | 112.25 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-07-14 09:35:00 | 111.75 | 2025-07-14 09:50:00 | 111.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:00:00 | 112.64 | 2025-07-15 10:10:00 | 113.07 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-15 10:00:00 | 112.64 | 2025-07-15 10:15:00 | 112.64 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 09:40:00 | 112.82 | 2025-07-16 09:50:00 | 112.58 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-17 09:30:00 | 114.26 | 2025-07-17 10:05:00 | 114.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-22 09:30:00 | 112.07 | 2025-07-22 09:40:00 | 111.70 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-22 09:30:00 | 112.07 | 2025-07-22 15:20:00 | 109.27 | TARGET_HIT | 0.50 | 2.50% |
| SELL | retest1 | 2025-07-23 11:15:00 | 108.99 | 2025-07-23 11:30:00 | 108.53 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-23 11:15:00 | 108.99 | 2025-07-23 12:10:00 | 108.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 11:15:00 | 109.03 | 2025-07-24 11:20:00 | 108.75 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-24 11:15:00 | 109.03 | 2025-07-24 13:20:00 | 109.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-30 10:30:00 | 109.89 | 2025-07-30 10:35:00 | 109.55 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-31 09:55:00 | 104.85 | 2025-07-31 10:15:00 | 104.05 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2025-07-31 09:55:00 | 104.85 | 2025-07-31 11:00:00 | 104.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:00:00 | 103.30 | 2025-08-06 10:15:00 | 103.56 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-19 09:40:00 | 106.41 | 2025-08-19 09:50:00 | 106.62 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-20 09:35:00 | 107.45 | 2025-08-20 09:45:00 | 107.64 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-08-21 10:30:00 | 107.88 | 2025-08-21 10:35:00 | 107.74 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-08-26 09:35:00 | 104.05 | 2025-08-26 09:45:00 | 104.24 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-29 09:45:00 | 100.75 | 2025-08-29 10:20:00 | 101.07 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-02 09:35:00 | 102.81 | 2025-09-02 09:45:00 | 103.13 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-09-02 09:35:00 | 102.81 | 2025-09-02 13:15:00 | 103.05 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2025-09-08 09:30:00 | 104.61 | 2025-09-08 09:45:00 | 104.37 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-10 10:10:00 | 105.91 | 2025-09-10 10:20:00 | 105.69 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-17 10:20:00 | 108.87 | 2025-09-17 10:25:00 | 109.14 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-09-17 10:20:00 | 108.87 | 2025-09-17 15:20:00 | 111.96 | TARGET_HIT | 0.50 | 2.84% |
| BUY | retest1 | 2025-09-18 10:00:00 | 113.13 | 2025-09-18 10:05:00 | 112.75 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-19 09:50:00 | 113.45 | 2025-09-19 10:00:00 | 113.15 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-25 10:50:00 | 112.08 | 2025-09-25 11:00:00 | 111.78 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-26 09:35:00 | 109.39 | 2025-09-26 09:45:00 | 109.75 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-29 10:55:00 | 109.31 | 2025-09-29 11:05:00 | 109.66 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-29 10:55:00 | 109.31 | 2025-09-29 11:25:00 | 109.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-30 09:50:00 | 111.77 | 2025-09-30 10:05:00 | 111.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-01 11:15:00 | 112.20 | 2025-10-01 11:45:00 | 111.71 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-01 11:15:00 | 112.20 | 2025-10-01 12:35:00 | 112.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 11:10:00 | 113.84 | 2025-10-06 11:30:00 | 114.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-08 09:50:00 | 112.99 | 2025-10-08 10:00:00 | 113.31 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-09 10:15:00 | 114.04 | 2025-10-09 10:25:00 | 113.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-10 09:30:00 | 115.34 | 2025-10-10 09:35:00 | 115.10 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-14 09:40:00 | 116.14 | 2025-10-14 09:55:00 | 116.40 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-15 11:05:00 | 116.49 | 2025-10-15 11:55:00 | 116.83 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-15 11:05:00 | 116.49 | 2025-10-15 14:20:00 | 116.49 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 10:40:00 | 115.98 | 2025-10-20 10:55:00 | 116.54 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-20 10:40:00 | 115.98 | 2025-10-20 15:20:00 | 118.14 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2025-10-23 09:35:00 | 119.59 | 2025-10-23 09:40:00 | 119.21 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-24 09:55:00 | 118.46 | 2025-10-24 10:15:00 | 118.15 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-29 10:30:00 | 122.26 | 2025-10-29 10:35:00 | 121.93 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-30 09:55:00 | 120.14 | 2025-10-30 10:05:00 | 120.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-31 09:30:00 | 121.58 | 2025-10-31 09:40:00 | 121.22 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-11 10:20:00 | 120.27 | 2025-11-11 10:25:00 | 120.56 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-12 09:45:00 | 123.25 | 2025-11-12 09:50:00 | 122.99 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-13 10:05:00 | 123.12 | 2025-11-13 10:10:00 | 122.84 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-14 09:30:00 | 122.40 | 2025-11-14 09:35:00 | 122.13 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-17 09:30:00 | 124.09 | 2025-11-17 09:35:00 | 123.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-20 10:55:00 | 124.65 | 2025-11-20 11:05:00 | 124.91 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-21 10:30:00 | 122.49 | 2025-11-21 10:45:00 | 122.13 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-21 10:30:00 | 122.49 | 2025-11-21 11:15:00 | 122.49 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 09:30:00 | 125.02 | 2025-11-26 10:00:00 | 125.56 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-26 09:30:00 | 125.02 | 2025-11-26 10:55:00 | 125.02 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:00:00 | 120.58 | 2025-12-05 10:05:00 | 120.26 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-08 11:05:00 | 118.72 | 2025-12-08 11:35:00 | 118.31 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-08 11:05:00 | 118.72 | 2025-12-08 15:20:00 | 115.98 | TARGET_HIT | 0.50 | 2.31% |
| BUY | retest1 | 2025-12-10 09:30:00 | 118.68 | 2025-12-10 09:40:00 | 119.09 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-10 09:30:00 | 118.68 | 2025-12-10 09:50:00 | 118.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-11 09:35:00 | 116.52 | 2025-12-11 10:05:00 | 116.90 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-16 11:00:00 | 117.68 | 2025-12-16 11:15:00 | 117.37 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-16 11:00:00 | 117.68 | 2025-12-16 15:20:00 | 117.06 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-17 09:40:00 | 118.54 | 2025-12-17 10:10:00 | 118.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-22 09:45:00 | 120.83 | 2025-12-22 10:15:00 | 121.18 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-22 09:45:00 | 120.83 | 2025-12-22 10:30:00 | 120.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 10:40:00 | 120.26 | 2025-12-26 11:05:00 | 119.96 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-26 10:40:00 | 120.26 | 2025-12-26 12:05:00 | 120.17 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2026-01-01 10:20:00 | 124.15 | 2026-01-01 12:35:00 | 123.82 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-07 11:15:00 | 126.71 | 2026-01-07 12:00:00 | 126.47 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-08 11:00:00 | 124.07 | 2026-01-08 11:10:00 | 123.62 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-08 11:00:00 | 124.07 | 2026-01-08 15:20:00 | 122.71 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2026-01-09 10:05:00 | 124.05 | 2026-01-09 10:45:00 | 124.65 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-01-09 10:05:00 | 124.05 | 2026-01-09 10:55:00 | 124.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-13 11:05:00 | 123.10 | 2026-01-13 12:25:00 | 123.39 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-14 10:55:00 | 126.18 | 2026-01-14 11:05:00 | 126.67 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-14 10:55:00 | 126.18 | 2026-01-14 11:30:00 | 126.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 09:35:00 | 133.98 | 2026-01-19 09:40:00 | 133.51 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-21 10:55:00 | 123.65 | 2026-01-21 11:00:00 | 124.15 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-22 09:30:00 | 126.20 | 2026-01-22 09:50:00 | 125.79 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-29 09:45:00 | 126.05 | 2026-01-29 09:50:00 | 125.67 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-30 10:55:00 | 125.12 | 2026-01-30 11:15:00 | 124.77 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-06 10:25:00 | 122.62 | 2026-02-06 10:40:00 | 122.09 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-06 10:25:00 | 122.62 | 2026-02-06 11:40:00 | 122.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 11:15:00 | 121.41 | 2026-02-12 12:00:00 | 121.04 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-12 11:15:00 | 121.41 | 2026-02-12 12:30:00 | 121.41 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:15:00 | 122.13 | 2026-02-17 10:20:00 | 122.66 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 10:15:00 | 122.13 | 2026-02-17 15:20:00 | 124.91 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2026-02-18 09:30:00 | 126.45 | 2026-02-18 09:40:00 | 127.01 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-18 09:30:00 | 126.45 | 2026-02-18 15:20:00 | 128.15 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2026-02-25 09:45:00 | 130.53 | 2026-02-25 10:15:00 | 130.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-11 10:40:00 | 117.38 | 2026-03-11 11:30:00 | 116.98 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-11 10:40:00 | 117.38 | 2026-03-11 15:20:00 | 115.81 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2026-03-12 09:50:00 | 115.55 | 2026-03-12 10:15:00 | 115.08 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-13 09:40:00 | 114.34 | 2026-03-13 09:55:00 | 113.72 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 09:40:00 | 114.34 | 2026-03-13 11:30:00 | 114.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 09:45:00 | 109.98 | 2026-04-08 14:30:00 | 110.80 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-04-08 09:45:00 | 109.98 | 2026-04-08 15:20:00 | 111.13 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2026-04-15 09:35:00 | 112.86 | 2026-04-15 09:40:00 | 113.27 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-16 10:55:00 | 114.62 | 2026-04-16 11:25:00 | 114.37 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-17 10:05:00 | 114.32 | 2026-04-17 10:30:00 | 114.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 09:30:00 | 114.69 | 2026-04-21 09:40:00 | 114.43 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-22 10:50:00 | 115.08 | 2026-04-22 11:20:00 | 115.49 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-22 10:50:00 | 115.08 | 2026-04-22 11:45:00 | 115.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:55:00 | 112.68 | 2026-04-28 11:05:00 | 112.35 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-30 10:10:00 | 108.85 | 2026-04-30 10:45:00 | 108.32 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-30 10:10:00 | 108.85 | 2026-04-30 12:10:00 | 108.85 | STOP_HIT | 0.50 | 0.00% |
