# Bajaj Finance Ltd. (BAJFINANCE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 954.50
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 8
- **Avg / median % per leg:** 0.33% / 0.30%
- **Sum % (uncompounded):** 7.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 8 | 57.1% | 3 | 6 | 5 | 0.29% | 4.0% |
| BUY @ 2nd Alert (retest1) | 14 | 8 | 57.1% | 3 | 6 | 5 | 0.29% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.41% | 3.3% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.41% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 12 | 54.5% | 4 | 10 | 8 | 0.33% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:50:00 | 968.05 | 974.54 | 0.00 | ORB-short ORB[974.00,985.15] vol=1.5x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 964.94 | 972.06 | 0.00 | T1 1.5R @ 964.94 |
| Stop hit — per-position SL triggered | 2026-02-10 10:35:00 | 968.05 | 970.02 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 971.00 | 966.09 | 0.00 | ORB-long ORB[964.20,968.20] vol=1.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:40:00 | 973.98 | 966.83 | 0.00 | T1 1.5R @ 973.98 |
| Stop hit — per-position SL triggered | 2026-02-11 10:50:00 | 971.00 | 967.72 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:45:00 | 975.10 | 971.91 | 0.00 | ORB-long ORB[965.00,972.45] vol=1.8x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:10:00 | 978.02 | 973.37 | 0.00 | T1 1.5R @ 978.02 |
| Target hit | 2026-02-12 15:20:00 | 999.90 | 990.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:10:00 | 1005.90 | 1003.19 | 0.00 | ORB-long ORB[992.80,1004.00] vol=1.8x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-02-13 10:20:00 | 1003.13 | 1003.58 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 1020.35 | 1017.25 | 0.00 | ORB-long ORB[1008.75,1019.90] vol=2.4x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:20:00 | 1024.37 | 1017.94 | 0.00 | T1 1.5R @ 1024.37 |
| Target hit | 2026-02-20 15:15:00 | 1027.20 | 1027.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-02-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:50:00 | 1013.00 | 1016.60 | 0.00 | ORB-short ORB[1015.50,1028.75] vol=1.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2026-02-24 09:55:00 | 1016.50 | 1016.46 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:10:00 | 1023.50 | 1020.88 | 0.00 | ORB-long ORB[1015.75,1022.45] vol=1.6x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:15:00 | 1026.94 | 1021.46 | 0.00 | T1 1.5R @ 1026.94 |
| Stop hit — per-position SL triggered | 2026-02-26 11:50:00 | 1023.50 | 1023.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:15:00 | 916.05 | 926.52 | 0.00 | ORB-short ORB[932.20,941.25] vol=4.5x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:45:00 | 912.62 | 924.08 | 0.00 | T1 1.5R @ 912.62 |
| Target hit | 2026-03-11 15:20:00 | 890.80 | 905.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 867.30 | 873.13 | 0.00 | ORB-short ORB[871.05,881.00] vol=2.2x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:25:00 | 863.25 | 872.44 | 0.00 | T1 1.5R @ 863.25 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 867.30 | 869.66 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 921.25 | 918.75 | 0.00 | ORB-long ORB[912.30,919.50] vol=4.4x ATR=4.10 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 917.15 | 920.37 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 928.25 | 925.42 | 0.00 | ORB-long ORB[919.00,927.95] vol=1.9x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 925.85 | 926.70 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 908.95 | 917.70 | 0.00 | ORB-short ORB[916.15,925.00] vol=1.6x ATR=2.82 |
| Stop hit — per-position SL triggered | 2026-04-27 10:40:00 | 911.77 | 917.53 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 929.45 | 926.55 | 0.00 | ORB-long ORB[916.60,925.95] vol=4.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-04-29 11:30:00 | 926.72 | 926.96 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 952.65 | 947.45 | 0.00 | ORB-long ORB[938.10,951.85] vol=1.7x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:35:00 | 958.94 | 950.93 | 0.00 | T1 1.5R @ 958.94 |
| Target hit | 2026-05-04 12:40:00 | 953.55 | 954.21 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:50:00 | 968.05 | 2026-02-10 10:10:00 | 964.94 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 09:50:00 | 968.05 | 2026-02-10 10:35:00 | 968.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:35:00 | 971.00 | 2026-02-11 10:40:00 | 973.98 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-11 10:35:00 | 971.00 | 2026-02-11 10:50:00 | 971.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 09:45:00 | 975.10 | 2026-02-12 10:10:00 | 978.02 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-12 09:45:00 | 975.10 | 2026-02-12 15:20:00 | 999.90 | TARGET_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2026-02-13 10:10:00 | 1005.90 | 2026-02-13 10:20:00 | 1003.13 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-20 11:00:00 | 1020.35 | 2026-02-20 11:20:00 | 1024.37 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-20 11:00:00 | 1020.35 | 2026-02-20 15:15:00 | 1027.20 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-24 09:50:00 | 1013.00 | 2026-02-24 09:55:00 | 1016.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-26 11:10:00 | 1023.50 | 2026-02-26 11:15:00 | 1026.94 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-26 11:10:00 | 1023.50 | 2026-02-26 11:50:00 | 1023.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:15:00 | 916.05 | 2026-03-11 11:45:00 | 912.62 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-11 11:15:00 | 916.05 | 2026-03-11 15:20:00 | 890.80 | TARGET_HIT | 0.50 | 2.76% |
| SELL | retest1 | 2026-03-17 11:15:00 | 867.30 | 2026-03-17 11:25:00 | 863.25 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-17 11:15:00 | 867.30 | 2026-03-17 13:15:00 | 867.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:45:00 | 921.25 | 2026-04-10 10:05:00 | 917.15 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-21 09:45:00 | 928.25 | 2026-04-21 10:25:00 | 925.85 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-27 10:35:00 | 908.95 | 2026-04-27 10:40:00 | 911.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-29 11:05:00 | 929.45 | 2026-04-29 11:30:00 | 926.72 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-04 09:35:00 | 952.65 | 2026-05-04 10:35:00 | 958.94 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-05-04 09:35:00 | 952.65 | 2026-05-04 12:40:00 | 953.55 | TARGET_HIT | 0.50 | 0.09% |
