# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 890.85
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 6
- **Avg / median % per leg:** 0.25% / 0.16%
- **Sum % (uncompounded):** 4.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.02% | 0.2% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.02% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.54% | 4.3% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.54% | 4.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 9 | 50.0% | 3 | 9 | 6 | 0.25% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 844.55 | 840.87 | 0.00 | ORB-long ORB[835.00,840.25] vol=1.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 842.48 | 841.63 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 848.90 | 844.73 | 0.00 | ORB-long ORB[840.00,846.45] vol=1.7x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-02-20 10:05:00 | 845.66 | 845.59 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 866.85 | 860.67 | 0.00 | ORB-long ORB[855.00,862.65] vol=3.2x ATR=3.38 |
| Stop hit — per-position SL triggered | 2026-02-23 10:10:00 | 863.47 | 862.23 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 844.90 | 846.75 | 0.00 | ORB-short ORB[845.05,853.55] vol=1.7x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-02-27 11:10:00 | 847.38 | 845.75 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:05:00 | 778.50 | 783.35 | 0.00 | ORB-short ORB[783.70,793.70] vol=5.0x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 780.95 | 782.70 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:35:00 | 718.95 | 724.01 | 0.00 | ORB-short ORB[720.00,729.35] vol=2.8x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:50:00 | 714.63 | 721.48 | 0.00 | T1 1.5R @ 714.63 |
| Target hit | 2026-03-27 15:20:00 | 700.45 | 707.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 764.00 | 757.73 | 0.00 | ORB-long ORB[751.00,759.90] vol=3.1x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:45:00 | 769.00 | 761.90 | 0.00 | T1 1.5R @ 769.00 |
| Stop hit — per-position SL triggered | 2026-04-15 10:00:00 | 764.00 | 762.47 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 777.00 | 771.83 | 0.00 | ORB-long ORB[763.00,771.90] vol=1.5x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 780.37 | 774.10 | 0.00 | T1 1.5R @ 780.37 |
| Stop hit — per-position SL triggered | 2026-04-21 11:40:00 | 777.00 | 776.89 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 890.35 | 884.47 | 0.00 | ORB-long ORB[879.00,887.10] vol=1.7x ATR=4.24 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 886.11 | 885.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 883.60 | 887.79 | 0.00 | ORB-short ORB[885.20,896.85] vol=1.8x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 12:20:00 | 878.89 | 884.40 | 0.00 | T1 1.5R @ 878.89 |
| Target hit | 2026-05-06 15:00:00 | 882.20 | 880.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2026-05-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:20:00 | 903.25 | 893.34 | 0.00 | ORB-long ORB[887.00,896.00] vol=4.4x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:25:00 | 908.67 | 898.02 | 0.00 | T1 1.5R @ 908.67 |
| Stop hit — per-position SL triggered | 2026-05-07 10:35:00 | 903.25 | 899.12 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:10:00 | 895.05 | 896.81 | 0.00 | ORB-short ORB[897.40,904.70] vol=1.7x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:30:00 | 891.36 | 896.52 | 0.00 | T1 1.5R @ 891.36 |
| Target hit | 2026-05-08 15:20:00 | 889.50 | 892.96 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:55:00 | 844.55 | 2026-02-17 10:15:00 | 842.48 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-20 09:45:00 | 848.90 | 2026-02-20 10:05:00 | 845.66 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-23 09:45:00 | 866.85 | 2026-02-23 10:10:00 | 863.47 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-27 09:50:00 | 844.90 | 2026-02-27 11:10:00 | 847.38 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-13 11:05:00 | 778.50 | 2026-03-13 11:25:00 | 780.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-27 09:35:00 | 718.95 | 2026-03-27 09:50:00 | 714.63 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-27 09:35:00 | 718.95 | 2026-03-27 15:20:00 | 700.45 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2026-04-15 09:35:00 | 764.00 | 2026-04-15 09:45:00 | 769.00 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-15 09:35:00 | 764.00 | 2026-04-15 10:00:00 | 764.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:50:00 | 777.00 | 2026-04-21 10:05:00 | 780.37 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-21 09:50:00 | 777.00 | 2026-04-21 11:40:00 | 777.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:30:00 | 890.35 | 2026-05-05 09:40:00 | 886.11 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-05-06 09:50:00 | 883.60 | 2026-05-06 12:20:00 | 878.89 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-06 09:50:00 | 883.60 | 2026-05-06 15:00:00 | 882.20 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-05-07 10:20:00 | 903.25 | 2026-05-07 10:25:00 | 908.67 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-05-07 10:20:00 | 903.25 | 2026-05-07 10:35:00 | 903.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:10:00 | 895.05 | 2026-05-08 11:30:00 | 891.36 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-05-08 11:10:00 | 895.05 | 2026-05-08 15:20:00 | 889.50 | TARGET_HIT | 0.50 | 0.62% |
