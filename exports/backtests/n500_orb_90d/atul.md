# Atul Ltd. (ATUL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 7090.00
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 5 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 18
- **Target hits / Stop hits / Partials:** 5 / 18 / 9
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 6.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 4 | 10 | 4 | 0.22% | 4.0% |
| BUY @ 2nd Alert (retest1) | 18 | 8 | 44.4% | 4 | 10 | 4 | 0.22% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 6 | 42.9% | 1 | 8 | 5 | 0.19% | 2.7% |
| SELL @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 1 | 8 | 5 | 0.19% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.21% | 6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 6519.50 | 6473.20 | 0.00 | ORB-long ORB[6410.00,6499.50] vol=3.2x ATR=34.83 |
| Target hit | 2026-02-09 15:20:00 | 6530.00 | 6516.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:15:00 | 6535.00 | 6579.79 | 0.00 | ORB-short ORB[6570.00,6646.00] vol=3.4x ATR=19.11 |
| Stop hit — per-position SL triggered | 2026-02-11 11:20:00 | 6554.11 | 6578.67 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 6602.50 | 6571.65 | 0.00 | ORB-long ORB[6516.50,6574.50] vol=1.8x ATR=15.95 |
| Stop hit — per-position SL triggered | 2026-02-17 10:10:00 | 6586.55 | 6577.03 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 6564.00 | 6610.57 | 0.00 | ORB-short ORB[6617.00,6646.00] vol=1.6x ATR=10.43 |
| Stop hit — per-position SL triggered | 2026-02-23 11:40:00 | 6574.43 | 6603.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:35:00 | 6528.00 | 6475.50 | 0.00 | ORB-long ORB[6420.00,6500.00] vol=1.6x ATR=13.25 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 6514.75 | 6477.58 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 6583.00 | 6539.37 | 0.00 | ORB-long ORB[6486.00,6560.00] vol=4.7x ATR=20.74 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 6562.26 | 6551.67 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 6588.00 | 6560.18 | 0.00 | ORB-long ORB[6475.00,6549.50] vol=3.2x ATR=19.66 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 6568.34 | 6562.31 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:40:00 | 6384.50 | 6425.45 | 0.00 | ORB-short ORB[6410.50,6475.00] vol=2.0x ATR=21.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:55:00 | 6352.79 | 6414.18 | 0.00 | T1 1.5R @ 6352.79 |
| Stop hit — per-position SL triggered | 2026-03-04 10:10:00 | 6384.50 | 6409.10 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 6445.50 | 6484.27 | 0.00 | ORB-short ORB[6512.00,6600.00] vol=2.5x ATR=19.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:50:00 | 6415.85 | 6464.62 | 0.00 | T1 1.5R @ 6415.85 |
| Stop hit — per-position SL triggered | 2026-03-05 12:35:00 | 6445.50 | 6450.46 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:50:00 | 6495.00 | 6544.36 | 0.00 | ORB-short ORB[6500.00,6579.00] vol=2.3x ATR=20.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:05:00 | 6464.53 | 6526.82 | 0.00 | T1 1.5R @ 6464.53 |
| Target hit | 2026-03-06 15:20:00 | 6403.00 | 6473.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 6090.00 | 6148.02 | 0.00 | ORB-short ORB[6186.00,6256.00] vol=3.2x ATR=23.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:45:00 | 6055.44 | 6117.13 | 0.00 | T1 1.5R @ 6055.44 |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 6090.00 | 6061.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 6124.00 | 6075.70 | 0.00 | ORB-long ORB[6030.50,6109.50] vol=4.0x ATR=23.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:35:00 | 6159.40 | 6096.68 | 0.00 | T1 1.5R @ 6159.40 |
| Target hit | 2026-03-11 15:20:00 | 6213.50 | 6150.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:05:00 | 6110.50 | 6173.83 | 0.00 | ORB-short ORB[6195.00,6279.50] vol=1.9x ATR=18.45 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 6128.95 | 6165.64 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 6232.00 | 6249.36 | 0.00 | ORB-short ORB[6238.00,6306.50] vol=8.2x ATR=16.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:15:00 | 6206.52 | 6248.50 | 0.00 | T1 1.5R @ 6206.52 |
| Stop hit — per-position SL triggered | 2026-03-17 13:25:00 | 6232.00 | 6206.60 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-03-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:40:00 | 6330.00 | 6291.91 | 0.00 | ORB-long ORB[6250.00,6305.50] vol=3.3x ATR=23.72 |
| Stop hit — per-position SL triggered | 2026-03-20 09:45:00 | 6306.28 | 6293.06 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-03-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-23 09:45:00 | 6195.50 | 6158.86 | 0.00 | ORB-long ORB[6111.50,6192.00] vol=2.2x ATR=27.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:50:00 | 6236.36 | 6183.16 | 0.00 | T1 1.5R @ 6236.36 |
| Stop hit — per-position SL triggered | 2026-03-23 09:55:00 | 6195.50 | 6186.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 10:35:00 | 6447.00 | 6374.71 | 0.00 | ORB-long ORB[6364.50,6416.50] vol=1.7x ATR=30.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 10:55:00 | 6492.30 | 6401.34 | 0.00 | T1 1.5R @ 6492.30 |
| Stop hit — per-position SL triggered | 2026-04-01 11:20:00 | 6447.00 | 6408.19 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:40:00 | 6255.00 | 6202.19 | 0.00 | ORB-long ORB[6148.50,6239.00] vol=1.7x ATR=16.60 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 6238.40 | 6206.25 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:25:00 | 6360.00 | 6321.75 | 0.00 | ORB-long ORB[6271.50,6323.50] vol=2.2x ATR=19.68 |
| Target hit | 2026-04-08 15:20:00 | 6409.00 | 6352.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 6357.50 | 6383.08 | 0.00 | ORB-short ORB[6376.00,6417.50] vol=3.5x ATR=24.04 |
| Stop hit — per-position SL triggered | 2026-04-15 09:35:00 | 6381.54 | 6381.30 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 6813.00 | 6737.91 | 0.00 | ORB-long ORB[6680.00,6739.00] vol=2.5x ATR=23.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 6848.05 | 6767.76 | 0.00 | T1 1.5R @ 6848.05 |
| Target hit | 2026-04-28 14:35:00 | 6912.00 | 6916.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 7024.50 | 7007.27 | 0.00 | ORB-long ORB[6952.50,7018.00] vol=3.2x ATR=27.41 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 6997.09 | 7007.39 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 6985.50 | 6950.97 | 0.00 | ORB-long ORB[6923.00,6974.50] vol=3.0x ATR=13.86 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 6971.64 | 6951.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 6519.50 | 2026-02-09 15:20:00 | 6530.00 | TARGET_HIT | 1.00 | 0.16% |
| SELL | retest1 | 2026-02-11 11:15:00 | 6535.00 | 2026-02-11 11:20:00 | 6554.11 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 09:50:00 | 6602.50 | 2026-02-17 10:10:00 | 6586.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-23 10:55:00 | 6564.00 | 2026-02-23 11:40:00 | 6574.43 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-25 10:35:00 | 6528.00 | 2026-02-25 10:40:00 | 6514.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-26 09:35:00 | 6583.00 | 2026-02-26 10:20:00 | 6562.26 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-27 10:15:00 | 6588.00 | 2026-02-27 10:20:00 | 6568.34 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-04 09:40:00 | 6384.50 | 2026-03-04 09:55:00 | 6352.79 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-04 09:40:00 | 6384.50 | 2026-03-04 10:10:00 | 6384.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 11:05:00 | 6445.50 | 2026-03-05 11:50:00 | 6415.85 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-05 11:05:00 | 6445.50 | 2026-03-05 12:35:00 | 6445.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:50:00 | 6495.00 | 2026-03-06 11:05:00 | 6464.53 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-06 10:50:00 | 6495.00 | 2026-03-06 15:20:00 | 6403.00 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2026-03-10 11:10:00 | 6090.00 | 2026-03-10 11:45:00 | 6055.44 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-10 11:10:00 | 6090.00 | 2026-03-10 13:15:00 | 6090.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 11:10:00 | 6124.00 | 2026-03-11 11:35:00 | 6159.40 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-11 11:10:00 | 6124.00 | 2026-03-11 15:20:00 | 6213.50 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2026-03-13 11:05:00 | 6110.50 | 2026-03-13 11:25:00 | 6128.95 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-17 11:05:00 | 6232.00 | 2026-03-17 11:15:00 | 6206.52 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-17 11:05:00 | 6232.00 | 2026-03-17 13:25:00 | 6232.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 09:40:00 | 6330.00 | 2026-03-20 09:45:00 | 6306.28 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-23 09:45:00 | 6195.50 | 2026-03-23 09:50:00 | 6236.36 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-23 09:45:00 | 6195.50 | 2026-03-23 09:55:00 | 6195.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-01 10:35:00 | 6447.00 | 2026-04-01 10:55:00 | 6492.30 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-04-01 10:35:00 | 6447.00 | 2026-04-01 11:20:00 | 6447.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 10:40:00 | 6255.00 | 2026-04-07 10:50:00 | 6238.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-08 10:25:00 | 6360.00 | 2026-04-08 15:20:00 | 6409.00 | TARGET_HIT | 1.00 | 0.77% |
| SELL | retest1 | 2026-04-15 09:30:00 | 6357.50 | 2026-04-15 09:35:00 | 6381.54 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-28 10:10:00 | 6813.00 | 2026-04-28 10:15:00 | 6848.05 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-28 10:10:00 | 6813.00 | 2026-04-28 14:35:00 | 6912.00 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2026-05-05 09:45:00 | 7024.50 | 2026-05-05 10:00:00 | 6997.09 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-06 11:15:00 | 6985.50 | 2026-05-06 11:25:00 | 6971.64 | STOP_HIT | 1.00 | -0.20% |
