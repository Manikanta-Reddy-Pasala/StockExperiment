# Punjab National Bank (PNB)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-08-08 15:25:00 (4596 bars)
- **Last close:** 114.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 15
- **Target hits / Stop hits / Partials:** 6 / 15 / 13
- **Avg / median % per leg:** 0.34% / 0.34%
- **Sum % (uncompounded):** 11.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.23% | 2.5% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.23% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 13 | 56.5% | 4 | 10 | 9 | 0.40% | 9.1% |
| SELL @ 2nd Alert (retest1) | 23 | 13 | 56.5% | 4 | 10 | 9 | 0.40% | 9.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 34 | 19 | 55.9% | 6 | 15 | 13 | 0.34% | 11.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:05:00 | 124.35 | 125.57 | 0.00 | ORB-short ORB[124.70,126.10] vol=2.2x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 11:35:00 | 123.71 | 125.38 | 0.00 | T1 1.5R @ 123.71 |
| Stop hit — per-position SL triggered | 2024-05-16 14:45:00 | 124.35 | 124.57 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:20:00 | 127.85 | 126.60 | 0.00 | ORB-long ORB[125.80,127.10] vol=2.5x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-05-21 10:35:00 | 127.42 | 126.81 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:30:00 | 125.70 | 126.26 | 0.00 | ORB-short ORB[125.75,127.00] vol=1.5x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 09:40:00 | 125.05 | 125.96 | 0.00 | T1 1.5R @ 125.05 |
| Target hit | 2024-05-22 10:00:00 | 125.50 | 125.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-05-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:45:00 | 126.00 | 126.67 | 0.00 | ORB-short ORB[126.40,127.15] vol=2.6x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 09:50:00 | 125.40 | 126.48 | 0.00 | T1 1.5R @ 125.40 |
| Stop hit — per-position SL triggered | 2024-05-24 10:05:00 | 126.00 | 126.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:55:00 | 126.00 | 126.92 | 0.00 | ORB-short ORB[126.20,127.60] vol=1.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-05-27 10:00:00 | 126.48 | 126.88 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:30:00 | 128.70 | 127.93 | 0.00 | ORB-long ORB[126.95,128.30] vol=2.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 09:45:00 | 129.41 | 128.51 | 0.00 | T1 1.5R @ 129.41 |
| Target hit | 2024-05-29 12:25:00 | 129.35 | 129.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 127.25 | 127.94 | 0.00 | ORB-short ORB[127.70,129.25] vol=1.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:00:00 | 126.54 | 127.71 | 0.00 | T1 1.5R @ 126.54 |
| Stop hit — per-position SL triggered | 2024-05-31 11:55:00 | 127.25 | 127.20 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:00:00 | 127.48 | 126.93 | 0.00 | ORB-long ORB[126.00,127.20] vol=2.5x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-06-12 11:30:00 | 127.18 | 127.22 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:10:00 | 126.94 | 127.75 | 0.00 | ORB-short ORB[128.04,128.80] vol=2.2x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:20:00 | 126.54 | 127.63 | 0.00 | T1 1.5R @ 126.54 |
| Stop hit — per-position SL triggered | 2024-06-13 11:25:00 | 126.94 | 127.61 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:45:00 | 127.44 | 127.00 | 0.00 | ORB-long ORB[126.40,127.30] vol=1.8x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:20:00 | 127.95 | 127.33 | 0.00 | T1 1.5R @ 127.95 |
| Target hit | 2024-06-14 15:20:00 | 128.89 | 128.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2024-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:35:00 | 127.70 | 128.36 | 0.00 | ORB-short ORB[127.75,129.19] vol=2.1x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:40:00 | 127.03 | 128.14 | 0.00 | T1 1.5R @ 127.03 |
| Stop hit — per-position SL triggered | 2024-06-19 09:45:00 | 127.70 | 128.09 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 11:00:00 | 126.65 | 127.15 | 0.00 | ORB-short ORB[126.70,127.60] vol=1.7x ATR=0.24 |
| Stop hit — per-position SL triggered | 2024-06-21 11:10:00 | 126.89 | 127.13 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 125.05 | 123.86 | 0.00 | ORB-long ORB[123.37,124.32] vol=2.3x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:25:00 | 125.61 | 124.17 | 0.00 | T1 1.5R @ 125.61 |
| Stop hit — per-position SL triggered | 2024-06-26 12:35:00 | 125.05 | 124.57 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:15:00 | 122.90 | 123.64 | 0.00 | ORB-short ORB[123.56,124.50] vol=1.8x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:25:00 | 122.40 | 123.46 | 0.00 | T1 1.5R @ 122.40 |
| Target hit | 2024-06-27 15:20:00 | 118.76 | 120.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 122.27 | 123.02 | 0.00 | ORB-short ORB[122.31,123.95] vol=1.7x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-07-05 09:40:00 | 122.69 | 122.94 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:30:00 | 123.17 | 122.60 | 0.00 | ORB-long ORB[122.06,122.99] vol=1.8x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-07-08 09:40:00 | 122.82 | 122.67 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 122.53 | 122.22 | 0.00 | ORB-long ORB[121.64,122.45] vol=1.5x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:50:00 | 122.95 | 122.40 | 0.00 | T1 1.5R @ 122.95 |
| Stop hit — per-position SL triggered | 2024-07-09 10:20:00 | 122.53 | 122.76 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 121.07 | 121.65 | 0.00 | ORB-short ORB[121.28,122.95] vol=1.8x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:55:00 | 120.51 | 121.52 | 0.00 | T1 1.5R @ 120.51 |
| Target hit | 2024-07-10 15:20:00 | 119.20 | 119.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-07-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:00:00 | 117.95 | 118.76 | 0.00 | ORB-short ORB[118.00,119.55] vol=1.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:55:00 | 117.30 | 118.43 | 0.00 | T1 1.5R @ 117.30 |
| Target hit | 2024-07-19 15:20:00 | 116.44 | 117.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 124.54 | 125.03 | 0.00 | ORB-short ORB[124.80,126.00] vol=1.8x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-07-31 09:35:00 | 124.88 | 125.00 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:55:00 | 119.99 | 120.80 | 0.00 | ORB-short ORB[120.36,121.65] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-08-02 10:15:00 | 120.47 | 120.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:05:00 | 124.35 | 2024-05-16 11:35:00 | 123.71 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-16 11:05:00 | 124.35 | 2024-05-16 14:45:00 | 124.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-21 10:20:00 | 127.85 | 2024-05-21 10:35:00 | 127.42 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-22 09:30:00 | 125.70 | 2024-05-22 09:40:00 | 125.05 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-05-22 09:30:00 | 125.70 | 2024-05-22 10:00:00 | 125.50 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2024-05-24 09:45:00 | 126.00 | 2024-05-24 09:50:00 | 125.40 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-05-24 09:45:00 | 126.00 | 2024-05-24 10:05:00 | 126.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-27 09:55:00 | 126.00 | 2024-05-27 10:00:00 | 126.48 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-29 09:30:00 | 128.70 | 2024-05-29 09:45:00 | 129.41 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-05-29 09:30:00 | 128.70 | 2024-05-29 12:25:00 | 129.35 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-31 09:45:00 | 127.25 | 2024-05-31 10:00:00 | 126.54 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-05-31 09:45:00 | 127.25 | 2024-05-31 11:55:00 | 127.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 10:00:00 | 127.48 | 2024-06-12 11:30:00 | 127.18 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-06-13 11:10:00 | 126.94 | 2024-06-13 11:20:00 | 126.54 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-06-13 11:10:00 | 126.94 | 2024-06-13 11:25:00 | 126.94 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-14 09:45:00 | 127.44 | 2024-06-14 10:20:00 | 127.95 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-14 09:45:00 | 127.44 | 2024-06-14 15:20:00 | 128.89 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2024-06-19 09:35:00 | 127.70 | 2024-06-19 09:40:00 | 127.03 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-06-19 09:35:00 | 127.70 | 2024-06-19 09:45:00 | 127.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 11:00:00 | 126.65 | 2024-06-21 11:10:00 | 126.89 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-06-26 11:00:00 | 125.05 | 2024-06-26 11:25:00 | 125.61 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-26 11:00:00 | 125.05 | 2024-06-26 12:35:00 | 125.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:15:00 | 122.90 | 2024-06-27 10:25:00 | 122.40 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-27 10:15:00 | 122.90 | 2024-06-27 15:20:00 | 118.76 | TARGET_HIT | 0.50 | 3.37% |
| SELL | retest1 | 2024-07-05 09:30:00 | 122.27 | 2024-07-05 09:40:00 | 122.69 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-08 09:30:00 | 123.17 | 2024-07-08 09:40:00 | 122.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-09 09:30:00 | 122.53 | 2024-07-09 09:50:00 | 122.95 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-07-09 09:30:00 | 122.53 | 2024-07-09 10:20:00 | 122.53 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 09:45:00 | 121.07 | 2024-07-10 09:55:00 | 120.51 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-10 09:45:00 | 121.07 | 2024-07-10 15:20:00 | 119.20 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2024-07-19 10:00:00 | 117.95 | 2024-07-19 10:55:00 | 117.30 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-19 10:00:00 | 117.95 | 2024-07-19 15:20:00 | 116.44 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2024-07-31 09:30:00 | 124.54 | 2024-07-31 09:35:00 | 124.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-02 09:55:00 | 119.99 | 2024-08-02 10:15:00 | 120.47 | STOP_HIT | 1.00 | -0.40% |
