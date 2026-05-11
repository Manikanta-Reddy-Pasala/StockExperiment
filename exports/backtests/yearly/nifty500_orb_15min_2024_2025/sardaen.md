# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-12-06 15:25:00 (10683 bars)
- **Last close:** 471.50
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
| ENTRY1 | 25 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 5 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 20
- **Target hits / Stop hits / Partials:** 5 / 20 / 13
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 10.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.23% | 4.6% |
| BUY @ 2nd Alert (retest1) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.23% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 9 | 50.0% | 2 | 9 | 7 | 0.32% | 5.7% |
| SELL @ 2nd Alert (retest1) | 18 | 9 | 50.0% | 2 | 9 | 7 | 0.32% | 5.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 38 | 18 | 47.4% | 5 | 20 | 13 | 0.27% | 10.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:35:00 | 271.10 | 271.83 | 0.00 | ORB-short ORB[271.20,274.95] vol=3.5x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 09:50:00 | 269.09 | 271.51 | 0.00 | T1 1.5R @ 269.09 |
| Target hit | 2024-05-14 12:10:00 | 270.40 | 270.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2024-05-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:45:00 | 267.25 | 265.25 | 0.00 | ORB-long ORB[263.60,265.95] vol=3.4x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-05-17 10:50:00 | 266.29 | 265.40 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:20:00 | 223.35 | 220.31 | 0.00 | ORB-long ORB[217.80,220.70] vol=3.0x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:25:00 | 225.29 | 221.44 | 0.00 | T1 1.5R @ 225.29 |
| Stop hit — per-position SL triggered | 2024-06-07 10:30:00 | 223.35 | 221.83 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:50:00 | 229.35 | 227.36 | 0.00 | ORB-long ORB[225.89,228.90] vol=2.3x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-06-11 10:05:00 | 228.08 | 227.62 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:00:00 | 231.52 | 229.93 | 0.00 | ORB-long ORB[228.72,230.99] vol=3.1x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-06-13 10:05:00 | 230.62 | 229.97 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 241.77 | 239.35 | 0.00 | ORB-long ORB[236.41,239.69] vol=1.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-06-14 10:05:00 | 240.44 | 239.73 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:50:00 | 234.34 | 236.05 | 0.00 | ORB-short ORB[236.00,239.25] vol=2.1x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:00:00 | 232.47 | 235.52 | 0.00 | T1 1.5R @ 232.47 |
| Stop hit — per-position SL triggered | 2024-06-18 10:05:00 | 234.34 | 235.41 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:00:00 | 230.50 | 228.45 | 0.00 | ORB-long ORB[226.75,229.89] vol=2.1x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:10:00 | 232.09 | 229.13 | 0.00 | T1 1.5R @ 232.09 |
| Stop hit — per-position SL triggered | 2024-06-20 10:30:00 | 230.50 | 229.43 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:15:00 | 229.17 | 226.99 | 0.00 | ORB-long ORB[225.25,227.47] vol=1.9x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 11:20:00 | 230.24 | 227.25 | 0.00 | T1 1.5R @ 230.24 |
| Target hit | 2024-06-24 15:20:00 | 234.30 | 230.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-06-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 09:35:00 | 231.18 | 232.01 | 0.00 | ORB-short ORB[231.32,233.34] vol=2.4x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:55:00 | 229.61 | 231.09 | 0.00 | T1 1.5R @ 229.61 |
| Stop hit — per-position SL triggered | 2024-06-28 12:40:00 | 231.18 | 230.93 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:50:00 | 234.09 | 233.25 | 0.00 | ORB-long ORB[231.80,233.98] vol=5.2x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:55:00 | 235.89 | 235.29 | 0.00 | T1 1.5R @ 235.89 |
| Target hit | 2024-07-02 10:15:00 | 235.27 | 235.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:15:00 | 239.00 | 236.85 | 0.00 | ORB-long ORB[234.83,236.98] vol=6.0x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 12:20:00 | 240.59 | 238.42 | 0.00 | T1 1.5R @ 240.59 |
| Stop hit — per-position SL triggered | 2024-07-04 12:35:00 | 239.00 | 238.64 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:00:00 | 275.65 | 272.03 | 0.00 | ORB-long ORB[269.00,272.88] vol=3.0x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 273.39 | 274.14 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 11:15:00 | 278.00 | 280.08 | 0.00 | ORB-short ORB[280.25,282.85] vol=3.3x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-07-12 11:30:00 | 279.43 | 279.99 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 268.88 | 270.03 | 0.00 | ORB-short ORB[269.81,273.51] vol=2.4x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 270.08 | 270.13 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:30:00 | 281.05 | 279.98 | 0.00 | ORB-long ORB[277.11,280.90] vol=2.4x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-07-30 09:55:00 | 279.56 | 280.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 331.50 | 329.57 | 0.00 | ORB-long ORB[327.40,330.20] vol=1.9x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:00:00 | 334.51 | 334.47 | 0.00 | T1 1.5R @ 334.51 |
| Target hit | 2024-08-21 10:20:00 | 337.80 | 338.12 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2024-08-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:05:00 | 344.00 | 344.96 | 0.00 | ORB-short ORB[346.20,351.25] vol=10.7x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:15:00 | 341.29 | 344.81 | 0.00 | T1 1.5R @ 341.29 |
| Stop hit — per-position SL triggered | 2024-08-23 12:25:00 | 344.00 | 344.41 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:20:00 | 354.15 | 351.77 | 0.00 | ORB-long ORB[346.05,349.95] vol=3.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-08-29 10:25:00 | 352.36 | 351.88 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:00:00 | 350.55 | 355.07 | 0.00 | ORB-short ORB[355.80,359.90] vol=1.8x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 11:15:00 | 347.85 | 351.99 | 0.00 | T1 1.5R @ 347.85 |
| Stop hit — per-position SL triggered | 2024-09-06 11:40:00 | 350.55 | 351.65 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 503.90 | 506.62 | 0.00 | ORB-short ORB[505.80,511.00] vol=2.0x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:45:00 | 501.20 | 506.04 | 0.00 | T1 1.5R @ 501.20 |
| Stop hit — per-position SL triggered | 2024-10-16 11:55:00 | 503.90 | 505.84 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 11:15:00 | 420.30 | 424.30 | 0.00 | ORB-short ORB[422.75,428.40] vol=2.4x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:40:00 | 417.76 | 423.85 | 0.00 | T1 1.5R @ 417.76 |
| Target hit | 2024-11-19 15:20:00 | 409.80 | 416.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:10:00 | 426.00 | 431.29 | 0.00 | ORB-short ORB[430.05,434.85] vol=1.8x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-11-29 10:30:00 | 428.06 | 429.56 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-12-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 11:05:00 | 479.15 | 484.56 | 0.00 | ORB-short ORB[480.50,487.40] vol=1.5x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-12-05 11:15:00 | 481.58 | 484.44 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:10:00 | 485.80 | 479.34 | 0.00 | ORB-long ORB[475.15,480.40] vol=6.6x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-12-06 10:15:00 | 482.25 | 479.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:35:00 | 271.10 | 2024-05-14 09:50:00 | 269.09 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-05-14 09:35:00 | 271.10 | 2024-05-14 12:10:00 | 270.40 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-05-17 10:45:00 | 267.25 | 2024-05-17 10:50:00 | 266.29 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-07 10:20:00 | 223.35 | 2024-06-07 10:25:00 | 225.29 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-06-07 10:20:00 | 223.35 | 2024-06-07 10:30:00 | 223.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 09:50:00 | 229.35 | 2024-06-11 10:05:00 | 228.08 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-06-13 10:00:00 | 231.52 | 2024-06-13 10:05:00 | 230.62 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-14 09:55:00 | 241.77 | 2024-06-14 10:05:00 | 240.44 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-06-18 09:50:00 | 234.34 | 2024-06-18 10:00:00 | 232.47 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2024-06-18 09:50:00 | 234.34 | 2024-06-18 10:05:00 | 234.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 10:00:00 | 230.50 | 2024-06-20 10:10:00 | 232.09 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-06-20 10:00:00 | 230.50 | 2024-06-20 10:30:00 | 230.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 11:15:00 | 229.17 | 2024-06-24 11:20:00 | 230.24 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-24 11:15:00 | 229.17 | 2024-06-24 15:20:00 | 234.30 | TARGET_HIT | 0.50 | 2.24% |
| SELL | retest1 | 2024-06-28 09:35:00 | 231.18 | 2024-06-28 11:55:00 | 229.61 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-06-28 09:35:00 | 231.18 | 2024-06-28 12:40:00 | 231.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 09:50:00 | 234.09 | 2024-07-02 09:55:00 | 235.89 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-07-02 09:50:00 | 234.09 | 2024-07-02 10:15:00 | 235.27 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-04 10:15:00 | 239.00 | 2024-07-04 12:20:00 | 240.59 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-07-04 10:15:00 | 239.00 | 2024-07-04 12:35:00 | 239.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 10:00:00 | 275.65 | 2024-07-11 10:15:00 | 273.39 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest1 | 2024-07-12 11:15:00 | 278.00 | 2024-07-12 11:30:00 | 279.43 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-07-23 11:15:00 | 268.88 | 2024-07-23 11:20:00 | 270.08 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-30 09:30:00 | 281.05 | 2024-07-30 09:55:00 | 279.56 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-08-21 09:30:00 | 331.50 | 2024-08-21 10:00:00 | 334.51 | PARTIAL | 0.50 | 0.91% |
| BUY | retest1 | 2024-08-21 09:30:00 | 331.50 | 2024-08-21 10:20:00 | 337.80 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2024-08-23 10:05:00 | 344.00 | 2024-08-23 10:15:00 | 341.29 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2024-08-23 10:05:00 | 344.00 | 2024-08-23 12:25:00 | 344.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 10:20:00 | 354.15 | 2024-08-29 10:25:00 | 352.36 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-09-06 10:00:00 | 350.55 | 2024-09-06 11:15:00 | 347.85 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-09-06 10:00:00 | 350.55 | 2024-09-06 11:40:00 | 350.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 11:15:00 | 503.90 | 2024-10-16 11:45:00 | 501.20 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-16 11:15:00 | 503.90 | 2024-10-16 11:55:00 | 503.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-19 11:15:00 | 420.30 | 2024-11-19 11:40:00 | 417.76 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-11-19 11:15:00 | 420.30 | 2024-11-19 15:20:00 | 409.80 | TARGET_HIT | 0.50 | 2.50% |
| SELL | retest1 | 2024-11-29 10:10:00 | 426.00 | 2024-11-29 10:30:00 | 428.06 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-12-05 11:05:00 | 479.15 | 2024-12-05 11:15:00 | 481.58 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-12-06 10:10:00 | 485.80 | 2024-12-06 10:15:00 | 482.25 | STOP_HIT | 1.00 | -0.73% |
