# IndusInd Bank Ltd. (INDUSINDBK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 948.45
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 7
- **Avg / median % per leg:** 0.18% / 0.16%
- **Sum % (uncompounded):** 3.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.23% | 2.5% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.23% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.12% | 0.9% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.12% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 10 | 52.6% | 3 | 9 | 7 | 0.18% | 3.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 904.00 | 910.40 | 0.00 | ORB-short ORB[913.00,922.00] vol=3.3x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-02-13 11:25:00 | 906.79 | 909.60 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 944.25 | 938.19 | 0.00 | ORB-long ORB[928.20,940.75] vol=4.9x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 940.84 | 941.11 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:15:00 | 922.50 | 924.60 | 0.00 | ORB-short ORB[923.10,934.65] vol=4.6x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:40:00 | 916.91 | 923.67 | 0.00 | T1 1.5R @ 916.91 |
| Target hit | 2026-02-23 15:10:00 | 921.00 | 920.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 949.85 | 945.09 | 0.00 | ORB-long ORB[936.30,946.80] vol=4.5x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:35:00 | 953.49 | 948.18 | 0.00 | T1 1.5R @ 953.49 |
| Stop hit — per-position SL triggered | 2026-02-26 10:05:00 | 949.85 | 956.02 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 946.80 | 940.94 | 0.00 | ORB-long ORB[930.05,942.95] vol=1.7x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 944.25 | 941.19 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 896.80 | 890.79 | 0.00 | ORB-long ORB[887.65,894.80] vol=2.3x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:20:00 | 902.18 | 893.75 | 0.00 | T1 1.5R @ 902.18 |
| Stop hit — per-position SL triggered | 2026-03-10 11:50:00 | 896.80 | 894.51 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 813.80 | 816.07 | 0.00 | ORB-short ORB[814.10,825.00] vol=1.5x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:25:00 | 809.49 | 815.13 | 0.00 | T1 1.5R @ 809.49 |
| Stop hit — per-position SL triggered | 2026-03-17 12:40:00 | 813.80 | 814.34 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:30:00 | 786.70 | 795.74 | 0.00 | ORB-short ORB[798.85,809.00] vol=1.6x ATR=3.73 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 790.43 | 794.45 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:35:00 | 815.50 | 812.14 | 0.00 | ORB-long ORB[802.95,814.70] vol=1.5x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 09:50:00 | 821.26 | 816.35 | 0.00 | T1 1.5R @ 821.26 |
| Target hit | 2026-03-25 12:50:00 | 820.60 | 821.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 827.70 | 819.02 | 0.00 | ORB-long ORB[810.00,822.00] vol=2.1x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:40:00 | 834.64 | 821.90 | 0.00 | T1 1.5R @ 834.64 |
| Target hit | 2026-04-08 13:05:00 | 830.60 | 830.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 856.35 | 860.59 | 0.00 | ORB-short ORB[858.45,868.00] vol=2.2x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:55:00 | 852.67 | 858.88 | 0.00 | T1 1.5R @ 852.67 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 856.35 | 858.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 938.45 | 934.41 | 0.00 | ORB-long ORB[924.00,937.90] vol=1.5x ATR=3.67 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 934.78 | 934.91 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 11:00:00 | 904.00 | 2026-02-13 11:25:00 | 906.79 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-17 09:35:00 | 944.25 | 2026-02-17 09:50:00 | 940.84 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-23 10:15:00 | 922.50 | 2026-02-23 10:40:00 | 916.91 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-02-23 10:15:00 | 922.50 | 2026-02-23 15:10:00 | 921.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-02-26 09:30:00 | 949.85 | 2026-02-26 09:35:00 | 953.49 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-26 09:30:00 | 949.85 | 2026-02-26 10:05:00 | 949.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 11:05:00 | 946.80 | 2026-03-05 11:15:00 | 944.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-10 10:40:00 | 896.80 | 2026-03-10 11:20:00 | 902.18 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-10 10:40:00 | 896.80 | 2026-03-10 11:50:00 | 896.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 11:05:00 | 813.80 | 2026-03-17 11:25:00 | 809.49 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-17 11:05:00 | 813.80 | 2026-03-17 12:40:00 | 813.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-23 10:30:00 | 786.70 | 2026-03-23 11:05:00 | 790.43 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-25 09:35:00 | 815.50 | 2026-03-25 09:50:00 | 821.26 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-25 09:35:00 | 815.50 | 2026-03-25 12:50:00 | 820.60 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-08 09:35:00 | 827.70 | 2026-04-08 09:40:00 | 834.64 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2026-04-08 09:35:00 | 827.70 | 2026-04-08 13:05:00 | 830.60 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-24 09:30:00 | 856.35 | 2026-04-24 09:55:00 | 852.67 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-24 09:30:00 | 856.35 | 2026-04-24 10:00:00 | 856.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:30:00 | 938.45 | 2026-05-04 09:50:00 | 934.78 | STOP_HIT | 1.00 | -0.39% |
