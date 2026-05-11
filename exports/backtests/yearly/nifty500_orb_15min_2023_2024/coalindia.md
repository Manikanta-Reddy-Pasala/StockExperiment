# Coal India Ltd. (COALINDIA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53854 bars)
- **Last close:** 456.55
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
| PARTIAL | 41 |
| TARGET_HIT | 18 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 80
- **Target hits / Stop hits / Partials:** 18 / 80 / 41
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 20.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 83 | 31 | 37.3% | 10 | 52 | 21 | 0.12% | 10.2% |
| BUY @ 2nd Alert (retest1) | 83 | 31 | 37.3% | 10 | 52 | 21 | 0.12% | 10.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 28 | 50.0% | 8 | 28 | 20 | 0.19% | 10.6% |
| SELL @ 2nd Alert (retest1) | 56 | 28 | 50.0% | 8 | 28 | 20 | 0.19% | 10.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 139 | 59 | 42.4% | 18 | 80 | 41 | 0.15% | 20.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:50:00 | 236.25 | 235.58 | 0.00 | ORB-long ORB[233.85,235.50] vol=4.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-05-15 09:55:00 | 235.71 | 235.73 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:45:00 | 239.50 | 238.30 | 0.00 | ORB-long ORB[237.65,239.20] vol=2.5x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-05-16 09:55:00 | 238.98 | 238.78 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 10:00:00 | 240.25 | 241.24 | 0.00 | ORB-short ORB[240.60,242.20] vol=2.0x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 10:40:00 | 239.64 | 240.92 | 0.00 | T1 1.5R @ 239.64 |
| Target hit | 2023-05-25 15:00:00 | 240.00 | 239.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2023-05-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:20:00 | 241.70 | 240.61 | 0.00 | ORB-long ORB[239.10,240.95] vol=1.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-05-26 11:50:00 | 241.18 | 241.06 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 10:45:00 | 243.05 | 242.12 | 0.00 | ORB-long ORB[240.80,242.30] vol=2.1x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 11:35:00 | 243.69 | 242.56 | 0.00 | T1 1.5R @ 243.69 |
| Target hit | 2023-05-29 15:20:00 | 245.95 | 245.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2023-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:30:00 | 229.60 | 228.77 | 0.00 | ORB-long ORB[228.10,229.05] vol=1.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-06-02 10:05:00 | 229.02 | 228.95 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:50:00 | 228.00 | 229.20 | 0.00 | ORB-short ORB[229.70,230.65] vol=2.1x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 11:00:00 | 227.50 | 229.05 | 0.00 | T1 1.5R @ 227.50 |
| Stop hit — per-position SL triggered | 2023-06-06 13:40:00 | 228.00 | 228.18 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:05:00 | 230.10 | 229.29 | 0.00 | ORB-long ORB[228.10,229.95] vol=1.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-06-07 10:20:00 | 229.71 | 229.38 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 11:00:00 | 228.15 | 229.01 | 0.00 | ORB-short ORB[228.65,229.65] vol=4.2x ATR=0.35 |
| Stop hit — per-position SL triggered | 2023-06-15 11:05:00 | 228.50 | 228.92 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:05:00 | 227.40 | 228.00 | 0.00 | ORB-short ORB[227.55,228.80] vol=1.7x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 13:15:00 | 226.95 | 227.73 | 0.00 | T1 1.5R @ 226.95 |
| Stop hit — per-position SL triggered | 2023-06-19 15:20:00 | 227.60 | 227.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2023-06-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:45:00 | 229.40 | 229.07 | 0.00 | ORB-long ORB[227.80,229.25] vol=5.2x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-06-22 12:00:00 | 229.08 | 229.15 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:40:00 | 225.10 | 225.74 | 0.00 | ORB-short ORB[225.60,227.55] vol=2.2x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-06-23 09:50:00 | 225.55 | 225.73 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 09:30:00 | 224.30 | 225.25 | 0.00 | ORB-short ORB[224.50,226.80] vol=1.7x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-06-27 10:15:00 | 224.75 | 225.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 11:15:00 | 226.35 | 225.50 | 0.00 | ORB-long ORB[224.15,225.40] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-06-28 12:00:00 | 226.02 | 225.60 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 10:45:00 | 231.95 | 230.95 | 0.00 | ORB-long ORB[230.80,231.90] vol=2.5x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 10:50:00 | 232.61 | 231.28 | 0.00 | T1 1.5R @ 232.61 |
| Stop hit — per-position SL triggered | 2023-07-03 10:55:00 | 231.95 | 231.34 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:05:00 | 232.25 | 231.80 | 0.00 | ORB-long ORB[230.65,232.15] vol=1.7x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-07-05 11:10:00 | 231.83 | 232.06 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:45:00 | 234.55 | 233.18 | 0.00 | ORB-long ORB[231.00,232.70] vol=2.5x ATR=0.53 |
| Stop hit — per-position SL triggered | 2023-07-06 10:00:00 | 234.02 | 233.59 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 11:15:00 | 233.95 | 234.57 | 0.00 | ORB-short ORB[234.05,235.60] vol=3.2x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 12:10:00 | 233.34 | 234.26 | 0.00 | T1 1.5R @ 233.34 |
| Stop hit — per-position SL triggered | 2023-07-10 12:15:00 | 233.95 | 234.25 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:55:00 | 235.10 | 234.41 | 0.00 | ORB-long ORB[233.50,234.70] vol=3.3x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-07-11 11:10:00 | 234.62 | 234.60 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:35:00 | 236.60 | 236.29 | 0.00 | ORB-long ORB[234.35,235.50] vol=1.6x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-07-12 09:45:00 | 236.15 | 236.37 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:00:00 | 233.55 | 233.96 | 0.00 | ORB-short ORB[234.40,235.30] vol=1.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 10:10:00 | 232.97 | 233.51 | 0.00 | T1 1.5R @ 232.97 |
| Target hit | 2023-07-13 12:05:00 | 231.40 | 231.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — BUY (started 2023-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:40:00 | 231.05 | 230.34 | 0.00 | ORB-long ORB[229.40,230.90] vol=3.5x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 09:45:00 | 231.79 | 230.73 | 0.00 | T1 1.5R @ 231.79 |
| Stop hit — per-position SL triggered | 2023-07-18 10:05:00 | 231.05 | 231.11 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:40:00 | 229.45 | 228.64 | 0.00 | ORB-long ORB[228.35,229.10] vol=1.5x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-07-20 11:05:00 | 229.08 | 228.76 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 11:00:00 | 229.75 | 229.30 | 0.00 | ORB-long ORB[228.75,229.70] vol=9.0x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 11:10:00 | 230.24 | 229.51 | 0.00 | T1 1.5R @ 230.24 |
| Target hit | 2023-07-24 15:20:00 | 230.95 | 230.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2023-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-26 09:50:00 | 230.45 | 230.65 | 0.00 | ORB-short ORB[230.50,231.20] vol=2.3x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-07-26 10:05:00 | 230.84 | 230.62 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 10:45:00 | 230.00 | 230.99 | 0.00 | ORB-short ORB[230.70,231.90] vol=3.3x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 11:35:00 | 229.45 | 230.30 | 0.00 | T1 1.5R @ 229.45 |
| Stop hit — per-position SL triggered | 2023-07-27 11:50:00 | 230.00 | 230.28 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-07-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 10:20:00 | 227.45 | 228.19 | 0.00 | ORB-short ORB[227.90,229.60] vol=3.0x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-07-28 10:25:00 | 227.89 | 228.17 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 10:00:00 | 229.40 | 229.00 | 0.00 | ORB-long ORB[227.55,229.35] vol=3.2x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-07-31 10:05:00 | 228.94 | 229.01 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 10:15:00 | 232.50 | 231.12 | 0.00 | ORB-long ORB[228.95,230.50] vol=2.2x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 10:20:00 | 233.34 | 231.45 | 0.00 | T1 1.5R @ 233.34 |
| Target hit | 2023-08-01 15:20:00 | 240.75 | 237.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2023-08-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 10:10:00 | 233.35 | 233.38 | 0.00 | ORB-short ORB[233.55,235.00] vol=4.8x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 12:20:00 | 232.30 | 233.08 | 0.00 | T1 1.5R @ 232.30 |
| Target hit | 2023-08-03 15:20:00 | 230.10 | 231.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2023-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 11:00:00 | 232.50 | 234.63 | 0.00 | ORB-short ORB[234.15,236.00] vol=2.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-08-07 11:05:00 | 233.07 | 234.61 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 09:35:00 | 232.85 | 234.04 | 0.00 | ORB-short ORB[232.95,235.40] vol=2.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-08-08 09:40:00 | 233.43 | 233.85 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 09:45:00 | 234.85 | 233.65 | 0.00 | ORB-long ORB[231.80,234.45] vol=3.7x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-08-09 09:50:00 | 233.98 | 233.69 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 10:25:00 | 230.65 | 228.92 | 0.00 | ORB-long ORB[227.00,228.35] vol=1.5x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-08-21 10:30:00 | 230.19 | 228.99 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 10:35:00 | 231.50 | 230.89 | 0.00 | ORB-long ORB[230.00,231.40] vol=10.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-08-22 10:40:00 | 231.02 | 230.90 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-08-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:30:00 | 232.50 | 232.14 | 0.00 | ORB-long ORB[231.45,232.40] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-08-24 10:00:00 | 232.17 | 232.22 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:15:00 | 228.85 | 229.44 | 0.00 | ORB-short ORB[228.90,230.10] vol=1.9x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 10:35:00 | 228.23 | 229.32 | 0.00 | T1 1.5R @ 228.23 |
| Stop hit — per-position SL triggered | 2023-08-25 11:20:00 | 228.85 | 228.95 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-08-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 10:00:00 | 229.25 | 228.38 | 0.00 | ORB-long ORB[227.50,228.70] vol=2.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 14:50:00 | 229.90 | 229.03 | 0.00 | T1 1.5R @ 229.90 |
| Target hit | 2023-08-28 15:20:00 | 229.35 | 229.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2023-08-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 10:05:00 | 229.80 | 229.23 | 0.00 | ORB-long ORB[228.70,229.55] vol=5.5x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 12:45:00 | 230.37 | 229.71 | 0.00 | T1 1.5R @ 230.37 |
| Stop hit — per-position SL triggered | 2023-08-31 14:00:00 | 229.80 | 229.75 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 09:30:00 | 233.95 | 232.95 | 0.00 | ORB-long ORB[230.65,233.50] vol=2.0x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 09:40:00 | 234.63 | 233.63 | 0.00 | T1 1.5R @ 234.63 |
| Target hit | 2023-09-01 12:30:00 | 234.65 | 234.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 276.25 | 277.74 | 0.00 | ORB-short ORB[276.60,280.05] vol=1.8x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:35:00 | 275.25 | 277.07 | 0.00 | T1 1.5R @ 275.25 |
| Target hit | 2023-09-12 10:00:00 | 274.30 | 273.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — BUY (started 2023-09-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-22 11:10:00 | 284.55 | 282.10 | 0.00 | ORB-long ORB[279.40,283.20] vol=7.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-09-22 11:30:00 | 283.46 | 282.72 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:05:00 | 282.45 | 283.80 | 0.00 | ORB-short ORB[283.05,285.35] vol=1.8x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 10:25:00 | 281.19 | 283.54 | 0.00 | T1 1.5R @ 281.19 |
| Stop hit — per-position SL triggered | 2023-09-25 10:35:00 | 282.45 | 283.47 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 11:15:00 | 290.25 | 286.91 | 0.00 | ORB-long ORB[284.50,288.05] vol=5.1x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 11:20:00 | 291.42 | 287.85 | 0.00 | T1 1.5R @ 291.42 |
| Stop hit — per-position SL triggered | 2023-09-27 12:50:00 | 290.25 | 289.76 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 09:45:00 | 286.95 | 288.33 | 0.00 | ORB-short ORB[287.60,289.20] vol=1.7x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 10:25:00 | 285.73 | 287.66 | 0.00 | T1 1.5R @ 285.73 |
| Target hit | 2023-10-05 12:30:00 | 286.35 | 286.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — BUY (started 2023-10-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:30:00 | 306.65 | 304.59 | 0.00 | ORB-long ORB[301.90,303.95] vol=3.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-10-12 09:45:00 | 305.76 | 305.70 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:40:00 | 319.15 | 318.25 | 0.00 | ORB-long ORB[315.95,318.65] vol=2.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-10-18 10:25:00 | 318.24 | 318.46 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:35:00 | 312.00 | 313.84 | 0.00 | ORB-short ORB[314.40,315.70] vol=1.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 10:15:00 | 310.78 | 312.34 | 0.00 | T1 1.5R @ 310.78 |
| Stop hit — per-position SL triggered | 2023-10-19 11:00:00 | 312.00 | 311.62 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:10:00 | 309.85 | 310.89 | 0.00 | ORB-short ORB[310.60,312.85] vol=3.0x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-10-23 10:15:00 | 310.76 | 310.86 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:50:00 | 304.45 | 305.99 | 0.00 | ORB-short ORB[306.60,309.15] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2023-10-26 10:05:00 | 305.63 | 305.70 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 10:05:00 | 311.30 | 309.47 | 0.00 | ORB-long ORB[306.00,309.00] vol=1.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-10-27 10:10:00 | 310.34 | 309.57 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:30:00 | 313.40 | 314.64 | 0.00 | ORB-short ORB[314.25,316.80] vol=2.1x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 09:35:00 | 312.45 | 314.01 | 0.00 | T1 1.5R @ 312.45 |
| Stop hit — per-position SL triggered | 2023-10-31 09:40:00 | 313.40 | 313.92 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:05:00 | 312.25 | 313.57 | 0.00 | ORB-short ORB[313.20,315.80] vol=1.7x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 10:25:00 | 310.90 | 313.09 | 0.00 | T1 1.5R @ 310.90 |
| Target hit | 2023-11-01 15:20:00 | 307.05 | 309.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2023-11-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-02 09:35:00 | 307.80 | 309.29 | 0.00 | ORB-short ORB[308.70,310.70] vol=1.9x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-11-02 09:50:00 | 308.76 | 309.02 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:20:00 | 312.75 | 311.56 | 0.00 | ORB-long ORB[310.30,311.90] vol=1.9x ATR=0.68 |
| Stop hit — per-position SL triggered | 2023-11-03 10:25:00 | 312.07 | 311.59 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:50:00 | 317.15 | 316.33 | 0.00 | ORB-long ORB[315.20,316.95] vol=1.6x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 11:20:00 | 318.05 | 316.62 | 0.00 | T1 1.5R @ 318.05 |
| Stop hit — per-position SL triggered | 2023-11-08 12:20:00 | 317.15 | 316.80 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 09:30:00 | 321.75 | 320.19 | 0.00 | ORB-long ORB[316.95,321.30] vol=4.3x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-11-09 09:35:00 | 321.01 | 320.24 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:55:00 | 337.45 | 335.65 | 0.00 | ORB-long ORB[333.00,336.90] vol=1.5x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-11-21 10:05:00 | 335.97 | 335.91 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:35:00 | 336.90 | 335.91 | 0.00 | ORB-long ORB[334.65,336.00] vol=3.3x ATR=0.84 |
| Stop hit — per-position SL triggered | 2023-11-24 09:40:00 | 336.06 | 335.92 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 09:30:00 | 338.35 | 340.36 | 0.00 | ORB-short ORB[339.50,343.45] vol=1.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2023-11-29 09:35:00 | 339.30 | 340.26 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-11-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:50:00 | 340.40 | 343.05 | 0.00 | ORB-short ORB[342.60,345.00] vol=2.0x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 10:00:00 | 338.82 | 342.28 | 0.00 | T1 1.5R @ 338.82 |
| Stop hit — per-position SL triggered | 2023-11-30 10:10:00 | 340.40 | 341.85 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 10:05:00 | 348.35 | 345.60 | 0.00 | ORB-long ORB[342.45,346.00] vol=1.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2023-12-01 10:10:00 | 347.25 | 345.78 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:30:00 | 355.00 | 353.14 | 0.00 | ORB-long ORB[350.65,354.80] vol=2.1x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 09:40:00 | 357.27 | 354.99 | 0.00 | T1 1.5R @ 357.27 |
| Target hit | 2023-12-04 11:50:00 | 356.35 | 356.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2023-12-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 09:35:00 | 355.00 | 353.44 | 0.00 | ORB-long ORB[352.00,353.70] vol=1.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-12-08 09:40:00 | 354.13 | 353.79 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:30:00 | 356.75 | 355.49 | 0.00 | ORB-long ORB[351.00,356.15] vol=2.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2023-12-11 10:40:00 | 355.22 | 356.52 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 346.30 | 348.04 | 0.00 | ORB-short ORB[348.15,349.80] vol=2.0x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 09:55:00 | 344.65 | 347.24 | 0.00 | T1 1.5R @ 344.65 |
| Stop hit — per-position SL triggered | 2023-12-13 10:00:00 | 346.30 | 347.17 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:35:00 | 363.50 | 360.51 | 0.00 | ORB-long ORB[356.40,360.45] vol=5.9x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 11:10:00 | 366.16 | 363.04 | 0.00 | T1 1.5R @ 366.16 |
| Stop hit — per-position SL triggered | 2023-12-22 12:10:00 | 363.50 | 363.77 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:45:00 | 368.40 | 366.17 | 0.00 | ORB-long ORB[362.50,367.90] vol=1.8x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 09:55:00 | 370.49 | 368.01 | 0.00 | T1 1.5R @ 370.49 |
| Stop hit — per-position SL triggered | 2023-12-26 12:20:00 | 368.40 | 369.26 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:15:00 | 370.90 | 368.23 | 0.00 | ORB-long ORB[366.55,369.50] vol=2.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 10:20:00 | 372.73 | 369.68 | 0.00 | T1 1.5R @ 372.73 |
| Target hit | 2023-12-28 15:20:00 | 380.60 | 376.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2024-01-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:40:00 | 384.40 | 381.26 | 0.00 | ORB-long ORB[377.45,382.50] vol=3.2x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 09:55:00 | 386.48 | 382.80 | 0.00 | T1 1.5R @ 386.48 |
| Stop hit — per-position SL triggered | 2024-01-01 10:10:00 | 384.40 | 383.12 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:45:00 | 390.00 | 387.12 | 0.00 | ORB-long ORB[384.25,387.40] vol=5.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-01-02 09:55:00 | 388.55 | 387.28 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:00:00 | 389.70 | 387.54 | 0.00 | ORB-long ORB[385.60,388.00] vol=1.7x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-01-04 10:05:00 | 388.75 | 387.70 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 11:15:00 | 384.15 | 386.37 | 0.00 | ORB-short ORB[386.10,388.50] vol=2.1x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-01-05 11:30:00 | 385.16 | 386.28 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:30:00 | 459.20 | 456.64 | 0.00 | ORB-long ORB[452.65,458.20] vol=1.9x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-02-07 09:45:00 | 456.58 | 457.20 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-02-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:45:00 | 457.05 | 454.88 | 0.00 | ORB-long ORB[453.40,456.50] vol=2.0x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-02-08 10:05:00 | 455.36 | 455.75 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-02-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 09:40:00 | 473.40 | 470.53 | 0.00 | ORB-long ORB[467.55,471.95] vol=1.8x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 09:45:00 | 476.59 | 471.63 | 0.00 | T1 1.5R @ 476.59 |
| Target hit | 2024-02-15 12:30:00 | 473.60 | 474.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 77 — SELL (started 2024-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 11:10:00 | 442.85 | 445.88 | 0.00 | ORB-short ORB[444.50,449.60] vol=1.7x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-02-23 11:50:00 | 444.20 | 445.43 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 10:45:00 | 446.65 | 444.76 | 0.00 | ORB-long ORB[440.55,446.00] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-02-26 10:55:00 | 445.38 | 444.86 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 433.70 | 439.16 | 0.00 | ORB-short ORB[439.15,442.20] vol=4.0x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 435.08 | 438.62 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-03-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 10:25:00 | 455.00 | 451.90 | 0.00 | ORB-long ORB[450.35,454.10] vol=3.9x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 13:00:00 | 457.69 | 453.35 | 0.00 | T1 1.5R @ 457.69 |
| Stop hit — per-position SL triggered | 2024-03-04 13:45:00 | 455.00 | 453.63 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 09:30:00 | 458.80 | 457.22 | 0.00 | ORB-long ORB[454.00,458.45] vol=2.4x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 09:40:00 | 460.89 | 458.55 | 0.00 | T1 1.5R @ 460.89 |
| Stop hit — per-position SL triggered | 2024-03-05 09:45:00 | 458.80 | 458.62 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 460.40 | 462.19 | 0.00 | ORB-short ORB[461.35,464.95] vol=2.3x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:35:00 | 458.22 | 461.77 | 0.00 | T1 1.5R @ 458.22 |
| Target hit | 2024-03-06 12:35:00 | 455.60 | 454.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 83 — BUY (started 2024-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 09:35:00 | 422.55 | 420.30 | 0.00 | ORB-long ORB[416.85,422.45] vol=1.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-03-19 09:50:00 | 421.02 | 420.94 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:10:00 | 415.60 | 419.39 | 0.00 | ORB-short ORB[418.50,422.80] vol=1.6x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:45:00 | 412.72 | 418.06 | 0.00 | T1 1.5R @ 412.72 |
| Stop hit — per-position SL triggered | 2024-03-20 10:55:00 | 415.60 | 417.94 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-03-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:45:00 | 429.40 | 427.50 | 0.00 | ORB-long ORB[423.15,428.90] vol=2.4x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-03-21 09:50:00 | 427.66 | 427.54 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-03-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 11:05:00 | 433.10 | 431.96 | 0.00 | ORB-long ORB[428.55,432.35] vol=2.5x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-03-22 12:05:00 | 431.90 | 432.12 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-03-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 09:35:00 | 435.95 | 437.30 | 0.00 | ORB-short ORB[436.75,439.80] vol=2.4x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-03-27 09:45:00 | 437.26 | 437.21 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:30:00 | 444.95 | 443.13 | 0.00 | ORB-long ORB[440.90,444.40] vol=1.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 09:40:00 | 446.68 | 444.01 | 0.00 | T1 1.5R @ 446.68 |
| Stop hit — per-position SL triggered | 2024-04-03 09:55:00 | 444.95 | 444.41 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-04-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 10:45:00 | 453.05 | 449.06 | 0.00 | ORB-long ORB[447.55,451.00] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-04-04 10:50:00 | 451.54 | 449.23 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 09:30:00 | 447.20 | 448.49 | 0.00 | ORB-short ORB[447.35,450.45] vol=3.4x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 09:45:00 | 445.68 | 447.98 | 0.00 | T1 1.5R @ 445.68 |
| Stop hit — per-position SL triggered | 2024-04-08 10:10:00 | 447.20 | 447.24 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-09 09:45:00 | 446.55 | 447.93 | 0.00 | ORB-short ORB[447.00,450.30] vol=2.2x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 10:20:00 | 444.51 | 447.01 | 0.00 | T1 1.5R @ 444.51 |
| Target hit | 2024-04-09 15:20:00 | 440.45 | 441.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — BUY (started 2024-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 09:30:00 | 445.20 | 443.96 | 0.00 | ORB-long ORB[442.10,444.70] vol=1.8x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 09:35:00 | 447.05 | 444.56 | 0.00 | T1 1.5R @ 447.05 |
| Target hit | 2024-04-10 15:20:00 | 456.30 | 452.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 93 — BUY (started 2024-04-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 09:30:00 | 459.65 | 456.76 | 0.00 | ORB-long ORB[452.55,457.35] vol=2.2x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-04-12 09:40:00 | 458.07 | 458.05 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:30:00 | 454.00 | 450.77 | 0.00 | ORB-long ORB[446.35,453.00] vol=1.7x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-04-16 10:25:00 | 452.19 | 452.97 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 09:30:00 | 455.45 | 454.02 | 0.00 | ORB-long ORB[452.30,455.40] vol=1.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-04-18 09:40:00 | 454.22 | 454.40 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 09:30:00 | 442.20 | 440.71 | 0.00 | ORB-long ORB[438.00,441.85] vol=1.8x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-04-22 09:45:00 | 440.50 | 440.95 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 10:45:00 | 440.15 | 443.08 | 0.00 | ORB-short ORB[442.20,445.75] vol=1.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-04-23 11:25:00 | 441.18 | 442.48 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:50:00 | 445.70 | 444.39 | 0.00 | ORB-long ORB[441.75,444.50] vol=1.9x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 11:10:00 | 447.46 | 445.78 | 0.00 | T1 1.5R @ 447.46 |
| Target hit | 2024-04-25 15:20:00 | 453.25 | 449.17 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:50:00 | 236.25 | 2023-05-15 09:55:00 | 235.71 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-16 09:45:00 | 239.50 | 2023-05-16 09:55:00 | 238.98 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-05-25 10:00:00 | 240.25 | 2023-05-25 10:40:00 | 239.64 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-05-25 10:00:00 | 240.25 | 2023-05-25 15:00:00 | 240.00 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-05-26 10:20:00 | 241.70 | 2023-05-26 11:50:00 | 241.18 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-05-29 10:45:00 | 243.05 | 2023-05-29 11:35:00 | 243.69 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-05-29 10:45:00 | 243.05 | 2023-05-29 15:20:00 | 245.95 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2023-06-02 09:30:00 | 229.60 | 2023-06-02 10:05:00 | 229.02 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-06-06 10:50:00 | 228.00 | 2023-06-06 11:00:00 | 227.50 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-06-06 10:50:00 | 228.00 | 2023-06-06 13:40:00 | 228.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-07 10:05:00 | 230.10 | 2023-06-07 10:20:00 | 229.71 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-15 11:00:00 | 228.15 | 2023-06-15 11:05:00 | 228.50 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-19 11:05:00 | 227.40 | 2023-06-19 13:15:00 | 226.95 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-06-19 11:05:00 | 227.40 | 2023-06-19 15:20:00 | 227.60 | STOP_HIT | 0.50 | -0.09% |
| BUY | retest1 | 2023-06-22 09:45:00 | 229.40 | 2023-06-22 12:00:00 | 229.08 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-06-23 09:40:00 | 225.10 | 2023-06-23 09:50:00 | 225.55 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-27 09:30:00 | 224.30 | 2023-06-27 10:15:00 | 224.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-28 11:15:00 | 226.35 | 2023-06-28 12:00:00 | 226.02 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-07-03 10:45:00 | 231.95 | 2023-07-03 10:50:00 | 232.61 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-07-03 10:45:00 | 231.95 | 2023-07-03 10:55:00 | 231.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-05 10:05:00 | 232.25 | 2023-07-05 11:10:00 | 231.83 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-06 09:45:00 | 234.55 | 2023-07-06 10:00:00 | 234.02 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-07-10 11:15:00 | 233.95 | 2023-07-10 12:10:00 | 233.34 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-07-10 11:15:00 | 233.95 | 2023-07-10 12:15:00 | 233.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-11 10:55:00 | 235.10 | 2023-07-11 11:10:00 | 234.62 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-12 09:35:00 | 236.60 | 2023-07-12 09:45:00 | 236.15 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-07-13 10:00:00 | 233.55 | 2023-07-13 10:10:00 | 232.97 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-07-13 10:00:00 | 233.55 | 2023-07-13 12:05:00 | 231.40 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2023-07-18 09:40:00 | 231.05 | 2023-07-18 09:45:00 | 231.79 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-07-18 09:40:00 | 231.05 | 2023-07-18 10:05:00 | 231.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 10:40:00 | 229.45 | 2023-07-20 11:05:00 | 229.08 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-07-24 11:00:00 | 229.75 | 2023-07-24 11:10:00 | 230.24 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2023-07-24 11:00:00 | 229.75 | 2023-07-24 15:20:00 | 230.95 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2023-07-26 09:50:00 | 230.45 | 2023-07-26 10:05:00 | 230.84 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-27 10:45:00 | 230.00 | 2023-07-27 11:35:00 | 229.45 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-07-27 10:45:00 | 230.00 | 2023-07-27 11:50:00 | 230.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-28 10:20:00 | 227.45 | 2023-07-28 10:25:00 | 227.89 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-07-31 10:00:00 | 229.40 | 2023-07-31 10:05:00 | 228.94 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-01 10:15:00 | 232.50 | 2023-08-01 10:20:00 | 233.34 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-08-01 10:15:00 | 232.50 | 2023-08-01 15:20:00 | 240.75 | TARGET_HIT | 0.50 | 3.55% |
| SELL | retest1 | 2023-08-03 10:10:00 | 233.35 | 2023-08-03 12:20:00 | 232.30 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-08-03 10:10:00 | 233.35 | 2023-08-03 15:20:00 | 230.10 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2023-08-07 11:00:00 | 232.50 | 2023-08-07 11:05:00 | 233.07 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-08 09:35:00 | 232.85 | 2023-08-08 09:40:00 | 233.43 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-08-09 09:45:00 | 234.85 | 2023-08-09 09:50:00 | 233.98 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-08-21 10:25:00 | 230.65 | 2023-08-21 10:30:00 | 230.19 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-22 10:35:00 | 231.50 | 2023-08-22 10:40:00 | 231.02 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-24 09:30:00 | 232.50 | 2023-08-24 10:00:00 | 232.17 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-08-25 10:15:00 | 228.85 | 2023-08-25 10:35:00 | 228.23 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-25 10:15:00 | 228.85 | 2023-08-25 11:20:00 | 228.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-28 10:00:00 | 229.25 | 2023-08-28 14:50:00 | 229.90 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-08-28 10:00:00 | 229.25 | 2023-08-28 15:20:00 | 229.35 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2023-08-31 10:05:00 | 229.80 | 2023-08-31 12:45:00 | 230.37 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-08-31 10:05:00 | 229.80 | 2023-08-31 14:00:00 | 229.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-01 09:30:00 | 233.95 | 2023-09-01 09:40:00 | 234.63 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-09-01 09:30:00 | 233.95 | 2023-09-01 12:30:00 | 234.65 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2023-09-12 09:30:00 | 276.25 | 2023-09-12 09:35:00 | 275.25 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-12 09:30:00 | 276.25 | 2023-09-12 10:00:00 | 274.30 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2023-09-22 11:10:00 | 284.55 | 2023-09-22 11:30:00 | 283.46 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-25 10:05:00 | 282.45 | 2023-09-25 10:25:00 | 281.19 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-09-25 10:05:00 | 282.45 | 2023-09-25 10:35:00 | 282.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-27 11:15:00 | 290.25 | 2023-09-27 11:20:00 | 291.42 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-09-27 11:15:00 | 290.25 | 2023-09-27 12:50:00 | 290.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-05 09:45:00 | 286.95 | 2023-10-05 10:25:00 | 285.73 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-10-05 09:45:00 | 286.95 | 2023-10-05 12:30:00 | 286.35 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2023-10-12 09:30:00 | 306.65 | 2023-10-12 09:45:00 | 305.76 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-18 09:40:00 | 319.15 | 2023-10-18 10:25:00 | 318.24 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-19 09:35:00 | 312.00 | 2023-10-19 10:15:00 | 310.78 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-10-19 09:35:00 | 312.00 | 2023-10-19 11:00:00 | 312.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-23 10:10:00 | 309.85 | 2023-10-23 10:15:00 | 310.76 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-26 09:50:00 | 304.45 | 2023-10-26 10:05:00 | 305.63 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-10-27 10:05:00 | 311.30 | 2023-10-27 10:10:00 | 310.34 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-10-31 09:30:00 | 313.40 | 2023-10-31 09:35:00 | 312.45 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-10-31 09:30:00 | 313.40 | 2023-10-31 09:40:00 | 313.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-01 10:05:00 | 312.25 | 2023-11-01 10:25:00 | 310.90 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-01 10:05:00 | 312.25 | 2023-11-01 15:20:00 | 307.05 | TARGET_HIT | 0.50 | 1.67% |
| SELL | retest1 | 2023-11-02 09:35:00 | 307.80 | 2023-11-02 09:50:00 | 308.76 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-03 10:20:00 | 312.75 | 2023-11-03 10:25:00 | 312.07 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-08 10:50:00 | 317.15 | 2023-11-08 11:20:00 | 318.05 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-11-08 10:50:00 | 317.15 | 2023-11-08 12:20:00 | 317.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-09 09:30:00 | 321.75 | 2023-11-09 09:35:00 | 321.01 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-21 09:55:00 | 337.45 | 2023-11-21 10:05:00 | 335.97 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-11-24 09:35:00 | 336.90 | 2023-11-24 09:40:00 | 336.06 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-11-29 09:30:00 | 338.35 | 2023-11-29 09:35:00 | 339.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-11-30 09:50:00 | 340.40 | 2023-11-30 10:00:00 | 338.82 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-11-30 09:50:00 | 340.40 | 2023-11-30 10:10:00 | 340.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-01 10:05:00 | 348.35 | 2023-12-01 10:10:00 | 347.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-12-04 09:30:00 | 355.00 | 2023-12-04 09:40:00 | 357.27 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-12-04 09:30:00 | 355.00 | 2023-12-04 11:50:00 | 356.35 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2023-12-08 09:35:00 | 355.00 | 2023-12-08 09:40:00 | 354.13 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-12-11 09:30:00 | 356.75 | 2023-12-11 10:40:00 | 355.22 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-12-13 09:30:00 | 346.30 | 2023-12-13 09:55:00 | 344.65 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-12-13 09:30:00 | 346.30 | 2023-12-13 10:00:00 | 346.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-22 09:35:00 | 363.50 | 2023-12-22 11:10:00 | 366.16 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2023-12-22 09:35:00 | 363.50 | 2023-12-22 12:10:00 | 363.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 09:45:00 | 368.40 | 2023-12-26 09:55:00 | 370.49 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2023-12-26 09:45:00 | 368.40 | 2023-12-26 12:20:00 | 368.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-28 10:15:00 | 370.90 | 2023-12-28 10:20:00 | 372.73 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-12-28 10:15:00 | 370.90 | 2023-12-28 15:20:00 | 380.60 | TARGET_HIT | 0.50 | 2.62% |
| BUY | retest1 | 2024-01-01 09:40:00 | 384.40 | 2024-01-01 09:55:00 | 386.48 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-01-01 09:40:00 | 384.40 | 2024-01-01 10:10:00 | 384.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-02 09:45:00 | 390.00 | 2024-01-02 09:55:00 | 388.55 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-01-04 10:00:00 | 389.70 | 2024-01-04 10:05:00 | 388.75 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-05 11:15:00 | 384.15 | 2024-01-05 11:30:00 | 385.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-02-07 09:30:00 | 459.20 | 2024-02-07 09:45:00 | 456.58 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-02-08 09:45:00 | 457.05 | 2024-02-08 10:05:00 | 455.36 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-02-15 09:40:00 | 473.40 | 2024-02-15 09:45:00 | 476.59 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-02-15 09:40:00 | 473.40 | 2024-02-15 12:30:00 | 473.60 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2024-02-23 11:10:00 | 442.85 | 2024-02-23 11:50:00 | 444.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-26 10:45:00 | 446.65 | 2024-02-26 10:55:00 | 445.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-28 10:55:00 | 433.70 | 2024-02-28 11:00:00 | 435.08 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-03-04 10:25:00 | 455.00 | 2024-03-04 13:00:00 | 457.69 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-03-04 10:25:00 | 455.00 | 2024-03-04 13:45:00 | 455.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-05 09:30:00 | 458.80 | 2024-03-05 09:40:00 | 460.89 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-03-05 09:30:00 | 458.80 | 2024-03-05 09:45:00 | 458.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-06 09:30:00 | 460.40 | 2024-03-06 09:35:00 | 458.22 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-03-06 09:30:00 | 460.40 | 2024-03-06 12:35:00 | 455.60 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2024-03-19 09:35:00 | 422.55 | 2024-03-19 09:50:00 | 421.02 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-03-20 10:10:00 | 415.60 | 2024-03-20 10:45:00 | 412.72 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-03-20 10:10:00 | 415.60 | 2024-03-20 10:55:00 | 415.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 09:45:00 | 429.40 | 2024-03-21 09:50:00 | 427.66 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-03-22 11:05:00 | 433.10 | 2024-03-22 12:05:00 | 431.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-27 09:35:00 | 435.95 | 2024-03-27 09:45:00 | 437.26 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-03 09:30:00 | 444.95 | 2024-04-03 09:40:00 | 446.68 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-04-03 09:30:00 | 444.95 | 2024-04-03 09:55:00 | 444.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-04 10:45:00 | 453.05 | 2024-04-04 10:50:00 | 451.54 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-08 09:30:00 | 447.20 | 2024-04-08 09:45:00 | 445.68 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-04-08 09:30:00 | 447.20 | 2024-04-08 10:10:00 | 447.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-09 09:45:00 | 446.55 | 2024-04-09 10:20:00 | 444.51 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-04-09 09:45:00 | 446.55 | 2024-04-09 15:20:00 | 440.45 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2024-04-10 09:30:00 | 445.20 | 2024-04-10 09:35:00 | 447.05 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-04-10 09:30:00 | 445.20 | 2024-04-10 15:20:00 | 456.30 | TARGET_HIT | 0.50 | 2.49% |
| BUY | retest1 | 2024-04-12 09:30:00 | 459.65 | 2024-04-12 09:40:00 | 458.07 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-04-16 09:30:00 | 454.00 | 2024-04-16 10:25:00 | 452.19 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-04-18 09:30:00 | 455.45 | 2024-04-18 09:40:00 | 454.22 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-04-22 09:30:00 | 442.20 | 2024-04-22 09:45:00 | 440.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-04-23 10:45:00 | 440.15 | 2024-04-23 11:25:00 | 441.18 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-04-25 09:50:00 | 445.70 | 2024-04-25 11:10:00 | 447.46 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-04-25 09:50:00 | 445.70 | 2024-04-25 15:20:00 | 453.25 | TARGET_HIT | 0.50 | 1.69% |
