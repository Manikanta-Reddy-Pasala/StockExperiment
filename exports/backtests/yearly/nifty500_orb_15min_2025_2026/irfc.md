# Indian Railway Finance Corporation Ltd. (IRFC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 106.02
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 12 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 73
- **Target hits / Stop hits / Partials:** 12 / 73 / 32
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 7.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 20 | 37.7% | 6 | 33 | 14 | 0.10% | 5.3% |
| BUY @ 2nd Alert (retest1) | 53 | 20 | 37.7% | 6 | 33 | 14 | 0.10% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 64 | 24 | 37.5% | 6 | 40 | 18 | 0.03% | 2.2% |
| SELL @ 2nd Alert (retest1) | 64 | 24 | 37.5% | 6 | 40 | 18 | 0.03% | 2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 117 | 44 | 37.6% | 12 | 73 | 32 | 0.06% | 7.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:50:00 | 127.16 | 125.96 | 0.00 | ORB-long ORB[124.70,126.32] vol=2.2x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-05-14 11:05:00 | 126.81 | 126.16 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 10:05:00 | 135.23 | 136.36 | 0.00 | ORB-short ORB[136.11,137.19] vol=1.6x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-05-23 10:25:00 | 135.69 | 135.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 139.56 | 140.27 | 0.00 | ORB-short ORB[139.81,141.24] vol=1.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-05-29 10:00:00 | 139.91 | 140.02 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:35:00 | 139.98 | 140.66 | 0.00 | ORB-short ORB[140.00,141.40] vol=1.7x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:45:00 | 139.38 | 140.43 | 0.00 | T1 1.5R @ 139.38 |
| Target hit | 2025-05-30 13:30:00 | 139.36 | 139.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2025-06-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:40:00 | 140.18 | 139.23 | 0.00 | ORB-long ORB[138.21,139.55] vol=2.2x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:50:00 | 140.76 | 139.54 | 0.00 | T1 1.5R @ 140.76 |
| Stop hit — per-position SL triggered | 2025-06-02 09:55:00 | 140.18 | 139.59 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 144.19 | 143.11 | 0.00 | ORB-long ORB[141.92,143.22] vol=3.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-06-03 09:40:00 | 143.69 | 143.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:55:00 | 142.66 | 140.88 | 0.00 | ORB-long ORB[140.40,141.44] vol=2.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:05:00 | 143.43 | 141.52 | 0.00 | T1 1.5R @ 143.43 |
| Target hit | 2025-06-04 15:10:00 | 144.71 | 144.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2025-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:30:00 | 145.91 | 146.73 | 0.00 | ORB-short ORB[146.45,147.81] vol=3.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-06-10 09:35:00 | 146.29 | 146.67 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:05:00 | 147.01 | 146.29 | 0.00 | ORB-long ORB[145.13,146.60] vol=1.9x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 146.61 | 146.34 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:40:00 | 144.50 | 145.43 | 0.00 | ORB-short ORB[145.00,146.19] vol=1.9x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-06-12 09:50:00 | 145.02 | 145.30 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 136.62 | 137.91 | 0.00 | ORB-short ORB[137.33,139.17] vol=3.1x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-06-16 09:35:00 | 137.16 | 137.71 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:45:00 | 139.67 | 138.88 | 0.00 | ORB-long ORB[138.10,139.19] vol=1.5x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-06-24 11:30:00 | 139.23 | 139.06 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:30:00 | 140.35 | 139.69 | 0.00 | ORB-long ORB[138.70,140.22] vol=1.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 09:45:00 | 140.90 | 140.05 | 0.00 | T1 1.5R @ 140.90 |
| Stop hit — per-position SL triggered | 2025-06-25 09:50:00 | 140.35 | 140.11 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:35:00 | 141.56 | 140.86 | 0.00 | ORB-long ORB[140.15,141.25] vol=1.8x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 141.11 | 140.87 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 141.25 | 141.92 | 0.00 | ORB-short ORB[141.34,142.60] vol=1.7x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:05:00 | 140.73 | 141.61 | 0.00 | T1 1.5R @ 140.73 |
| Target hit | 2025-07-02 15:20:00 | 140.28 | 140.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:50:00 | 141.33 | 140.20 | 0.00 | ORB-long ORB[139.27,140.60] vol=2.0x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-07-03 09:55:00 | 140.89 | 140.27 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:55:00 | 137.91 | 138.61 | 0.00 | ORB-short ORB[138.08,139.20] vol=2.1x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:05:00 | 137.39 | 138.30 | 0.00 | T1 1.5R @ 137.39 |
| Stop hit — per-position SL triggered | 2025-07-08 14:55:00 | 137.91 | 137.92 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:55:00 | 138.03 | 138.40 | 0.00 | ORB-short ORB[138.20,139.00] vol=2.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-07-09 10:20:00 | 138.29 | 138.36 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:00:00 | 135.35 | 136.37 | 0.00 | ORB-short ORB[136.26,137.37] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 135.68 | 136.32 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:20:00 | 135.08 | 135.31 | 0.00 | ORB-short ORB[135.20,136.10] vol=1.5x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 12:00:00 | 134.66 | 135.16 | 0.00 | T1 1.5R @ 134.66 |
| Stop hit — per-position SL triggered | 2025-07-17 12:55:00 | 135.08 | 135.07 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 134.14 | 134.77 | 0.00 | ORB-short ORB[134.53,135.43] vol=1.8x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 134.38 | 134.51 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 11:10:00 | 132.59 | 133.75 | 0.00 | ORB-short ORB[133.89,135.50] vol=1.5x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 132.96 | 133.72 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 11:05:00 | 131.35 | 132.17 | 0.00 | ORB-short ORB[132.22,133.20] vol=1.9x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-07-30 11:30:00 | 131.62 | 132.09 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 09:30:00 | 129.46 | 128.59 | 0.00 | ORB-long ORB[127.41,129.05] vol=2.1x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-08-04 09:40:00 | 128.98 | 128.69 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:50:00 | 128.08 | 128.25 | 0.00 | ORB-short ORB[128.21,128.97] vol=1.5x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:00:00 | 127.60 | 128.20 | 0.00 | T1 1.5R @ 127.60 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 128.08 | 128.08 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 09:55:00 | 126.25 | 126.79 | 0.00 | ORB-short ORB[126.36,127.50] vol=1.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:25:00 | 125.67 | 126.49 | 0.00 | T1 1.5R @ 125.67 |
| Target hit | 2025-08-07 14:30:00 | 125.93 | 125.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:15:00 | 126.50 | 126.73 | 0.00 | ORB-short ORB[126.82,127.70] vol=2.2x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:50:00 | 126.11 | 126.67 | 0.00 | T1 1.5R @ 126.11 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 126.50 | 126.65 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:00:00 | 126.57 | 125.99 | 0.00 | ORB-long ORB[125.23,126.45] vol=2.4x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-08-18 10:45:00 | 126.26 | 126.13 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:40:00 | 124.31 | 124.85 | 0.00 | ORB-short ORB[124.61,125.55] vol=1.5x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-08-19 09:50:00 | 124.58 | 124.78 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 125.35 | 125.65 | 0.00 | ORB-short ORB[125.44,126.20] vol=1.8x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-08-22 09:35:00 | 125.60 | 125.62 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 122.56 | 123.17 | 0.00 | ORB-short ORB[122.77,124.61] vol=2.1x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:35:00 | 122.12 | 123.02 | 0.00 | T1 1.5R @ 122.12 |
| Stop hit — per-position SL triggered | 2025-08-26 10:00:00 | 122.56 | 122.76 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:45:00 | 117.55 | 118.52 | 0.00 | ORB-short ORB[118.20,119.38] vol=1.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-08-29 10:20:00 | 117.97 | 118.28 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 119.99 | 119.16 | 0.00 | ORB-long ORB[118.23,119.43] vol=2.6x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-09-01 10:05:00 | 119.65 | 119.28 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:45:00 | 122.75 | 121.54 | 0.00 | ORB-long ORB[120.61,121.64] vol=3.2x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:50:00 | 123.30 | 121.83 | 0.00 | T1 1.5R @ 123.30 |
| Stop hit — per-position SL triggered | 2025-09-02 09:55:00 | 122.75 | 121.92 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:35:00 | 124.33 | 124.07 | 0.00 | ORB-long ORB[123.57,124.30] vol=2.3x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:40:00 | 124.77 | 124.20 | 0.00 | T1 1.5R @ 124.77 |
| Target hit | 2025-09-08 10:50:00 | 124.67 | 124.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2025-09-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:40:00 | 125.54 | 124.75 | 0.00 | ORB-long ORB[123.79,125.20] vol=2.7x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-09-09 10:10:00 | 125.15 | 125.01 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 126.14 | 125.60 | 0.00 | ORB-long ORB[125.05,125.61] vol=3.6x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 09:45:00 | 126.49 | 126.01 | 0.00 | T1 1.5R @ 126.49 |
| Stop hit — per-position SL triggered | 2025-09-10 10:45:00 | 126.14 | 126.23 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:30:00 | 129.67 | 129.25 | 0.00 | ORB-long ORB[128.80,129.49] vol=1.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-09-17 09:40:00 | 129.37 | 129.28 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:30:00 | 130.35 | 129.91 | 0.00 | ORB-long ORB[129.50,130.15] vol=2.0x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 09:40:00 | 130.79 | 130.09 | 0.00 | T1 1.5R @ 130.79 |
| Stop hit — per-position SL triggered | 2025-09-18 09:45:00 | 130.35 | 130.12 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 127.87 | 128.39 | 0.00 | ORB-short ORB[128.10,128.93] vol=1.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-09-23 09:35:00 | 128.17 | 128.35 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 122.72 | 123.38 | 0.00 | ORB-short ORB[123.15,124.30] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-09-26 09:35:00 | 123.04 | 123.31 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 11:00:00 | 124.49 | 123.80 | 0.00 | ORB-long ORB[123.00,124.33] vol=2.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-09-29 11:25:00 | 124.16 | 123.85 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:30:00 | 124.72 | 124.20 | 0.00 | ORB-long ORB[123.50,124.50] vol=1.9x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:35:00 | 125.32 | 124.29 | 0.00 | T1 1.5R @ 125.32 |
| Stop hit — per-position SL triggered | 2025-10-01 11:05:00 | 124.72 | 124.43 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 09:45:00 | 125.06 | 125.47 | 0.00 | ORB-short ORB[125.20,126.25] vol=1.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-10-06 10:20:00 | 125.34 | 125.35 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 126.03 | 125.58 | 0.00 | ORB-long ORB[124.72,125.88] vol=2.1x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:00:00 | 126.44 | 125.80 | 0.00 | T1 1.5R @ 126.44 |
| Target hit | 2025-10-10 15:00:00 | 126.40 | 126.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 124.95 | 125.36 | 0.00 | ORB-short ORB[125.05,126.10] vol=3.4x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-10-13 09:40:00 | 125.28 | 125.33 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:35:00 | 125.40 | 125.99 | 0.00 | ORB-short ORB[125.50,126.48] vol=1.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:55:00 | 125.00 | 125.77 | 0.00 | T1 1.5R @ 125.00 |
| Stop hit — per-position SL triggered | 2025-10-14 10:20:00 | 125.40 | 125.64 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 123.72 | 124.18 | 0.00 | ORB-short ORB[123.80,125.17] vol=1.8x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-10-17 09:40:00 | 123.97 | 124.12 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:55:00 | 124.97 | 124.41 | 0.00 | ORB-long ORB[123.60,124.63] vol=1.8x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 124.60 | 124.49 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 123.82 | 123.24 | 0.00 | ORB-long ORB[122.64,123.35] vol=2.3x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:45:00 | 124.21 | 123.56 | 0.00 | T1 1.5R @ 124.21 |
| Target hit | 2025-10-29 15:20:00 | 125.20 | 124.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-11-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:45:00 | 122.58 | 122.98 | 0.00 | ORB-short ORB[123.01,123.50] vol=1.6x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-11-04 10:55:00 | 122.76 | 122.96 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 118.68 | 119.14 | 0.00 | ORB-short ORB[118.80,120.25] vol=2.0x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-11-07 09:40:00 | 118.98 | 119.12 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 121.91 | 121.68 | 0.00 | ORB-long ORB[121.30,121.90] vol=2.2x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-11-10 10:25:00 | 121.62 | 121.79 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 120.22 | 120.50 | 0.00 | ORB-short ORB[120.30,121.20] vol=1.7x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:05:00 | 119.85 | 120.29 | 0.00 | T1 1.5R @ 119.85 |
| Stop hit — per-position SL triggered | 2025-11-11 11:00:00 | 120.22 | 120.18 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:35:00 | 122.90 | 122.54 | 0.00 | ORB-long ORB[121.70,122.83] vol=3.2x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-11-12 10:00:00 | 122.60 | 122.67 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:45:00 | 119.95 | 120.26 | 0.00 | ORB-short ORB[120.00,120.85] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-11-19 10:10:00 | 120.17 | 120.22 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:35:00 | 119.77 | 120.01 | 0.00 | ORB-short ORB[119.80,120.40] vol=2.2x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:05:00 | 119.50 | 119.90 | 0.00 | T1 1.5R @ 119.50 |
| Target hit | 2025-11-21 12:20:00 | 119.66 | 119.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — SELL (started 2025-11-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:05:00 | 118.00 | 118.58 | 0.00 | ORB-short ORB[118.67,119.42] vol=2.3x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:30:00 | 117.67 | 118.49 | 0.00 | T1 1.5R @ 117.67 |
| Stop hit — per-position SL triggered | 2025-11-27 12:40:00 | 118.00 | 118.39 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 116.35 | 116.63 | 0.00 | ORB-short ORB[116.40,117.00] vol=2.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-12-02 10:05:00 | 116.57 | 116.53 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 116.22 | 116.88 | 0.00 | ORB-short ORB[116.35,117.60] vol=2.3x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-12-03 09:35:00 | 116.48 | 116.83 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:55:00 | 113.90 | 114.31 | 0.00 | ORB-short ORB[114.23,115.19] vol=1.5x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 114.12 | 114.31 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 112.95 | 113.84 | 0.00 | ORB-short ORB[114.05,114.94] vol=2.0x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:35:00 | 112.61 | 113.65 | 0.00 | T1 1.5R @ 112.61 |
| Target hit | 2025-12-08 15:20:00 | 111.41 | 112.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2025-12-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:50:00 | 109.95 | 110.61 | 0.00 | ORB-short ORB[110.56,111.99] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-12-09 09:55:00 | 110.34 | 110.57 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:40:00 | 114.90 | 114.43 | 0.00 | ORB-long ORB[113.52,114.49] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-12-10 09:45:00 | 114.59 | 114.45 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:20:00 | 113.35 | 112.47 | 0.00 | ORB-long ORB[111.53,112.75] vol=2.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-12-11 11:00:00 | 112.98 | 112.71 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:10:00 | 113.15 | 113.35 | 0.00 | ORB-short ORB[113.22,113.91] vol=1.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 10:20:00 | 112.79 | 113.30 | 0.00 | T1 1.5R @ 112.79 |
| Stop hit — per-position SL triggered | 2025-12-12 11:50:00 | 113.15 | 113.14 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:00:00 | 112.48 | 112.71 | 0.00 | ORB-short ORB[112.59,113.27] vol=1.8x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-12-16 10:05:00 | 112.67 | 112.71 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:30:00 | 110.41 | 110.64 | 0.00 | ORB-short ORB[110.51,111.02] vol=2.0x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:40:00 | 110.05 | 110.57 | 0.00 | T1 1.5R @ 110.05 |
| Target hit | 2025-12-18 10:10:00 | 110.33 | 110.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — BUY (started 2025-12-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:55:00 | 116.30 | 115.28 | 0.00 | ORB-long ORB[114.09,115.42] vol=3.7x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 10:05:00 | 116.95 | 115.85 | 0.00 | T1 1.5R @ 116.95 |
| Target hit | 2025-12-22 15:20:00 | 117.02 | 116.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2026-01-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:45:00 | 129.16 | 128.15 | 0.00 | ORB-long ORB[127.15,128.37] vol=4.2x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-01-06 09:55:00 | 128.68 | 128.30 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 126.97 | 128.01 | 0.00 | ORB-short ORB[127.42,129.00] vol=2.1x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-01-08 09:40:00 | 127.38 | 127.87 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:15:00 | 122.15 | 121.62 | 0.00 | ORB-long ORB[120.65,122.12] vol=1.6x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:25:00 | 122.66 | 121.75 | 0.00 | T1 1.5R @ 122.66 |
| Stop hit — per-position SL triggered | 2026-01-14 11:30:00 | 122.15 | 121.77 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:10:00 | 122.47 | 121.72 | 0.00 | ORB-long ORB[121.10,122.45] vol=2.1x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 121.97 | 121.75 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:50:00 | 116.80 | 116.12 | 0.00 | ORB-long ORB[115.18,116.70] vol=1.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:10:00 | 117.36 | 116.29 | 0.00 | T1 1.5R @ 117.36 |
| Target hit | 2026-01-28 15:20:00 | 120.53 | 118.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-01-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:05:00 | 120.10 | 119.16 | 0.00 | ORB-long ORB[118.25,119.90] vol=1.7x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:10:00 | 120.76 | 119.39 | 0.00 | T1 1.5R @ 120.76 |
| Stop hit — per-position SL triggered | 2026-01-30 10:20:00 | 120.10 | 119.50 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:55:00 | 116.12 | 115.43 | 0.00 | ORB-long ORB[114.51,115.72] vol=1.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-02-04 10:25:00 | 115.75 | 115.64 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:40:00 | 114.85 | 115.61 | 0.00 | ORB-short ORB[115.20,116.50] vol=1.5x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 11:00:00 | 114.31 | 115.20 | 0.00 | T1 1.5R @ 114.31 |
| Stop hit — per-position SL triggered | 2026-02-05 13:20:00 | 114.85 | 115.03 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 114.25 | 114.79 | 0.00 | ORB-short ORB[114.43,115.80] vol=1.5x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:05:00 | 113.84 | 114.46 | 0.00 | T1 1.5R @ 113.84 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 114.25 | 114.44 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 111.34 | 111.80 | 0.00 | ORB-short ORB[111.51,112.85] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 111.66 | 111.80 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 111.60 | 111.18 | 0.00 | ORB-long ORB[110.57,111.50] vol=1.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-02-16 09:35:00 | 111.31 | 111.21 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 112.61 | 113.16 | 0.00 | ORB-short ORB[112.85,113.75] vol=1.8x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:15:00 | 112.23 | 113.01 | 0.00 | T1 1.5R @ 112.23 |
| Stop hit — per-position SL triggered | 2026-02-18 11:10:00 | 112.61 | 112.86 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 104.30 | 103.19 | 0.00 | ORB-long ORB[102.41,103.64] vol=4.0x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-04-16 11:35:00 | 103.95 | 103.48 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-04-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:00:00 | 105.96 | 105.37 | 0.00 | ORB-long ORB[104.41,105.90] vol=2.4x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 105.69 | 105.45 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 106.31 | 106.04 | 0.00 | ORB-long ORB[105.25,106.30] vol=2.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 105.99 | 106.17 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 105.19 | 104.67 | 0.00 | ORB-long ORB[104.00,104.87] vol=3.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 104.80 | 104.75 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:50:00 | 127.16 | 2025-05-14 11:05:00 | 126.81 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-23 10:05:00 | 135.23 | 2025-05-23 10:25:00 | 135.69 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-05-29 09:30:00 | 139.56 | 2025-05-29 10:00:00 | 139.91 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-30 09:35:00 | 139.98 | 2025-05-30 09:45:00 | 139.38 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-05-30 09:35:00 | 139.98 | 2025-05-30 13:30:00 | 139.36 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-02 09:40:00 | 140.18 | 2025-06-02 09:50:00 | 140.76 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-02 09:40:00 | 140.18 | 2025-06-02 09:55:00 | 140.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-03 09:35:00 | 144.19 | 2025-06-03 09:40:00 | 143.69 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-04 10:55:00 | 142.66 | 2025-06-04 11:05:00 | 143.43 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-04 10:55:00 | 142.66 | 2025-06-04 15:10:00 | 144.71 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2025-06-10 09:30:00 | 145.91 | 2025-06-10 09:35:00 | 146.29 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-11 10:05:00 | 147.01 | 2025-06-11 10:15:00 | 146.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-12 09:40:00 | 144.50 | 2025-06-12 09:50:00 | 145.02 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-06-16 09:30:00 | 136.62 | 2025-06-16 09:35:00 | 137.16 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-24 10:45:00 | 139.67 | 2025-06-24 11:30:00 | 139.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-25 09:30:00 | 140.35 | 2025-06-25 09:45:00 | 140.90 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-25 09:30:00 | 140.35 | 2025-06-25 09:50:00 | 140.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:35:00 | 141.56 | 2025-06-27 09:40:00 | 141.11 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-02 09:40:00 | 141.25 | 2025-07-02 10:05:00 | 140.73 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-02 09:40:00 | 141.25 | 2025-07-02 15:20:00 | 140.28 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-07-03 09:50:00 | 141.33 | 2025-07-03 09:55:00 | 140.89 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-08 09:55:00 | 137.91 | 2025-07-08 11:05:00 | 137.39 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-08 09:55:00 | 137.91 | 2025-07-08 14:55:00 | 137.91 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-09 09:55:00 | 138.03 | 2025-07-09 10:20:00 | 138.29 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-11 11:00:00 | 135.35 | 2025-07-11 11:10:00 | 135.68 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-17 10:20:00 | 135.08 | 2025-07-17 12:00:00 | 134.66 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-17 10:20:00 | 135.08 | 2025-07-17 12:55:00 | 135.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:15:00 | 134.14 | 2025-07-18 11:15:00 | 134.38 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-25 11:10:00 | 132.59 | 2025-07-25 11:15:00 | 132.96 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-30 11:05:00 | 131.35 | 2025-07-30 11:30:00 | 131.62 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-08-04 09:30:00 | 129.46 | 2025-08-04 09:40:00 | 128.98 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-06 09:50:00 | 128.08 | 2025-08-06 10:00:00 | 127.60 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-06 09:50:00 | 128.08 | 2025-08-06 10:20:00 | 128.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 09:55:00 | 126.25 | 2025-08-07 11:25:00 | 125.67 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-08-07 09:55:00 | 126.25 | 2025-08-07 14:30:00 | 125.93 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-08-12 11:15:00 | 126.50 | 2025-08-12 11:50:00 | 126.11 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-08-12 11:15:00 | 126.50 | 2025-08-12 12:15:00 | 126.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-18 10:00:00 | 126.57 | 2025-08-18 10:45:00 | 126.26 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-19 09:40:00 | 124.31 | 2025-08-19 09:50:00 | 124.58 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-22 09:30:00 | 125.35 | 2025-08-22 09:35:00 | 125.60 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-26 09:30:00 | 122.56 | 2025-08-26 09:35:00 | 122.12 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-26 09:30:00 | 122.56 | 2025-08-26 10:00:00 | 122.56 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 09:45:00 | 117.55 | 2025-08-29 10:20:00 | 117.97 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-01 09:45:00 | 119.99 | 2025-09-01 10:05:00 | 119.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-02 09:45:00 | 122.75 | 2025-09-02 09:50:00 | 123.30 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-02 09:45:00 | 122.75 | 2025-09-02 09:55:00 | 122.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 09:35:00 | 124.33 | 2025-09-08 09:40:00 | 124.77 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-08 09:35:00 | 124.33 | 2025-09-08 10:50:00 | 124.67 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-09-09 09:40:00 | 125.54 | 2025-09-09 10:10:00 | 125.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-10 09:40:00 | 126.14 | 2025-09-10 09:45:00 | 126.49 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-09-10 09:40:00 | 126.14 | 2025-09-10 10:45:00 | 126.14 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 09:30:00 | 129.67 | 2025-09-17 09:40:00 | 129.37 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-18 09:30:00 | 130.35 | 2025-09-18 09:40:00 | 130.79 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-18 09:30:00 | 130.35 | 2025-09-18 09:45:00 | 130.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 09:30:00 | 127.87 | 2025-09-23 09:35:00 | 128.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-26 09:30:00 | 122.72 | 2025-09-26 09:35:00 | 123.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-29 11:00:00 | 124.49 | 2025-09-29 11:25:00 | 124.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-01 10:30:00 | 124.72 | 2025-10-01 10:35:00 | 125.32 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-01 10:30:00 | 124.72 | 2025-10-01 11:05:00 | 124.72 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 09:45:00 | 125.06 | 2025-10-06 10:20:00 | 125.34 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-10 09:40:00 | 126.03 | 2025-10-10 10:00:00 | 126.44 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-10 09:40:00 | 126.03 | 2025-10-10 15:00:00 | 126.40 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-13 09:30:00 | 124.95 | 2025-10-13 09:40:00 | 125.28 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-14 09:35:00 | 125.40 | 2025-10-14 09:55:00 | 125.00 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-14 09:35:00 | 125.40 | 2025-10-14 10:20:00 | 125.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 09:30:00 | 123.72 | 2025-10-17 09:40:00 | 123.97 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-20 09:55:00 | 124.97 | 2025-10-20 10:15:00 | 124.60 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-29 10:25:00 | 123.82 | 2025-10-29 10:45:00 | 124.21 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-29 10:25:00 | 123.82 | 2025-10-29 15:20:00 | 125.20 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2025-11-04 10:45:00 | 122.58 | 2025-11-04 10:55:00 | 122.76 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-11-07 09:35:00 | 118.68 | 2025-11-07 09:40:00 | 118.98 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-10 09:30:00 | 121.91 | 2025-11-10 10:25:00 | 121.62 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-11 09:30:00 | 120.22 | 2025-11-11 10:05:00 | 119.85 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-11 09:30:00 | 120.22 | 2025-11-11 11:00:00 | 120.22 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 09:35:00 | 122.90 | 2025-11-12 10:00:00 | 122.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-19 09:45:00 | 119.95 | 2025-11-19 10:10:00 | 120.17 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-21 09:35:00 | 119.77 | 2025-11-21 10:05:00 | 119.50 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-11-21 09:35:00 | 119.77 | 2025-11-21 12:20:00 | 119.66 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2025-11-27 11:05:00 | 118.00 | 2025-11-27 11:30:00 | 117.67 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-27 11:05:00 | 118.00 | 2025-11-27 12:40:00 | 118.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-02 09:30:00 | 116.35 | 2025-12-02 10:05:00 | 116.57 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-03 09:30:00 | 116.22 | 2025-12-03 09:35:00 | 116.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-05 09:55:00 | 113.90 | 2025-12-05 10:00:00 | 114.12 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-08 11:00:00 | 112.95 | 2025-12-08 11:35:00 | 112.61 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-08 11:00:00 | 112.95 | 2025-12-08 15:20:00 | 111.41 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2025-12-09 09:50:00 | 109.95 | 2025-12-09 09:55:00 | 110.34 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-12-10 09:40:00 | 114.90 | 2025-12-10 09:45:00 | 114.59 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-11 10:20:00 | 113.35 | 2025-12-11 11:00:00 | 112.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-12 10:10:00 | 113.15 | 2025-12-12 10:20:00 | 112.79 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-12 10:10:00 | 113.15 | 2025-12-12 11:50:00 | 113.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 10:00:00 | 112.48 | 2025-12-16 10:05:00 | 112.67 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-18 09:30:00 | 110.41 | 2025-12-18 09:40:00 | 110.05 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-18 09:30:00 | 110.41 | 2025-12-18 10:10:00 | 110.33 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2025-12-22 09:55:00 | 116.30 | 2025-12-22 10:05:00 | 116.95 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-12-22 09:55:00 | 116.30 | 2025-12-22 15:20:00 | 117.02 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2026-01-06 09:45:00 | 129.16 | 2026-01-06 09:55:00 | 128.68 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-01-08 09:30:00 | 126.97 | 2026-01-08 09:40:00 | 127.38 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-14 11:15:00 | 122.15 | 2026-01-14 11:25:00 | 122.66 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-14 11:15:00 | 122.15 | 2026-01-14 11:30:00 | 122.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 10:10:00 | 122.47 | 2026-01-19 10:15:00 | 121.97 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-01-28 10:50:00 | 116.80 | 2026-01-28 11:10:00 | 117.36 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-01-28 10:50:00 | 116.80 | 2026-01-28 15:20:00 | 120.53 | TARGET_HIT | 0.50 | 3.19% |
| BUY | retest1 | 2026-01-30 10:05:00 | 120.10 | 2026-01-30 10:10:00 | 120.76 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-01-30 10:05:00 | 120.10 | 2026-01-30 10:20:00 | 120.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-04 09:55:00 | 116.12 | 2026-02-04 10:25:00 | 115.75 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-05 09:40:00 | 114.85 | 2026-02-05 11:00:00 | 114.31 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-05 09:40:00 | 114.85 | 2026-02-05 13:20:00 | 114.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 09:30:00 | 114.25 | 2026-02-11 10:05:00 | 113.84 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-11 09:30:00 | 114.25 | 2026-02-11 10:10:00 | 114.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 09:35:00 | 111.34 | 2026-02-13 09:40:00 | 111.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-16 09:30:00 | 111.60 | 2026-02-16 09:35:00 | 111.31 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-18 10:00:00 | 112.61 | 2026-02-18 10:15:00 | 112.23 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-18 10:00:00 | 112.61 | 2026-02-18 11:10:00 | 112.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 11:00:00 | 104.30 | 2026-04-16 11:35:00 | 103.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-27 11:00:00 | 105.96 | 2026-04-27 11:25:00 | 105.69 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-28 09:45:00 | 106.31 | 2026-04-28 11:05:00 | 105.99 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:35:00 | 105.19 | 2026-05-05 09:45:00 | 104.80 | STOP_HIT | 1.00 | -0.37% |
