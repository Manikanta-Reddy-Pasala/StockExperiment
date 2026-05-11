# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-08-08 15:25:00 (4875 bars)
- **Last close:** 1059.00
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
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 15
- **Target hits / Stop hits / Partials:** 6 / 15 / 11
- **Avg / median % per leg:** 0.20% / 0.31%
- **Sum % (uncompounded):** 6.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 10 | 47.6% | 3 | 11 | 7 | 0.20% | 4.1% |
| BUY @ 2nd Alert (retest1) | 21 | 10 | 47.6% | 3 | 11 | 7 | 0.20% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.22% | 2.4% |
| SELL @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.22% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 32 | 17 | 53.1% | 6 | 15 | 11 | 0.20% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 966.20 | 960.73 | 0.00 | ORB-long ORB[952.00,962.05] vol=2.0x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 09:35:00 | 971.40 | 966.07 | 0.00 | T1 1.5R @ 971.40 |
| Target hit | 2025-05-23 10:25:00 | 972.40 | 973.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2025-05-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:20:00 | 994.75 | 990.09 | 0.00 | ORB-long ORB[984.95,993.45] vol=2.1x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-05-27 10:40:00 | 991.93 | 990.48 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:45:00 | 1006.65 | 995.61 | 0.00 | ORB-long ORB[985.00,996.90] vol=1.9x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:55:00 | 1013.40 | 999.37 | 0.00 | T1 1.5R @ 1013.40 |
| Stop hit — per-position SL triggered | 2025-06-02 10:00:00 | 1006.65 | 1000.19 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:50:00 | 1050.50 | 1054.55 | 0.00 | ORB-short ORB[1053.30,1063.70] vol=3.1x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-06-05 10:55:00 | 1053.67 | 1054.25 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 10:50:00 | 1014.40 | 1016.69 | 0.00 | ORB-short ORB[1014.80,1020.95] vol=2.5x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 11:25:00 | 1011.59 | 1016.13 | 0.00 | T1 1.5R @ 1011.59 |
| Target hit | 2025-06-10 15:20:00 | 1010.00 | 1012.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:35:00 | 1009.55 | 1013.70 | 0.00 | ORB-short ORB[1009.65,1020.05] vol=1.6x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 12:00:00 | 1006.03 | 1011.59 | 0.00 | T1 1.5R @ 1006.03 |
| Target hit | 2025-06-11 15:20:00 | 999.45 | 1006.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:40:00 | 990.40 | 994.58 | 0.00 | ORB-short ORB[992.00,1004.15] vol=2.5x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-06-12 09:50:00 | 993.18 | 994.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:40:00 | 965.00 | 958.10 | 0.00 | ORB-long ORB[952.00,964.60] vol=1.7x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-06-17 09:45:00 | 961.32 | 958.38 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:40:00 | 962.15 | 957.70 | 0.00 | ORB-long ORB[950.80,958.00] vol=3.0x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:45:00 | 966.56 | 958.85 | 0.00 | T1 1.5R @ 966.56 |
| Stop hit — per-position SL triggered | 2025-06-19 10:05:00 | 962.15 | 961.91 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:25:00 | 963.90 | 956.58 | 0.00 | ORB-long ORB[953.35,959.55] vol=1.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-06-24 10:30:00 | 961.35 | 956.90 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:50:00 | 977.05 | 967.90 | 0.00 | ORB-long ORB[960.00,968.20] vol=1.6x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:35:00 | 982.27 | 973.26 | 0.00 | T1 1.5R @ 982.27 |
| Stop hit — per-position SL triggered | 2025-06-25 10:40:00 | 977.05 | 973.53 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 976.45 | 971.86 | 0.00 | ORB-long ORB[966.00,975.65] vol=1.8x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:35:00 | 981.25 | 973.55 | 0.00 | T1 1.5R @ 981.25 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 976.45 | 974.40 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:00:00 | 961.70 | 964.50 | 0.00 | ORB-short ORB[964.85,971.10] vol=2.2x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:10:00 | 958.67 | 963.31 | 0.00 | T1 1.5R @ 958.67 |
| Stop hit — per-position SL triggered | 2025-07-01 10:40:00 | 961.70 | 962.05 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:10:00 | 968.50 | 962.84 | 0.00 | ORB-long ORB[956.40,966.25] vol=2.0x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 966.20 | 963.06 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:45:00 | 985.60 | 979.26 | 0.00 | ORB-long ORB[975.00,985.00] vol=2.5x ATR=2.92 |
| Stop hit — per-position SL triggered | 2025-07-04 11:00:00 | 982.68 | 980.16 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:10:00 | 965.00 | 967.52 | 0.00 | ORB-short ORB[967.00,978.40] vol=2.8x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-07-08 12:25:00 | 967.08 | 966.52 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 981.60 | 975.94 | 0.00 | ORB-long ORB[968.00,976.90] vol=1.5x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 09:40:00 | 986.13 | 981.23 | 0.00 | T1 1.5R @ 986.13 |
| Target hit | 2025-07-09 13:10:00 | 997.15 | 997.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2025-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:10:00 | 1006.70 | 999.27 | 0.00 | ORB-long ORB[991.45,1006.15] vol=2.1x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 10:35:00 | 1011.53 | 1002.98 | 0.00 | T1 1.5R @ 1011.53 |
| Target hit | 2025-07-10 12:20:00 | 1009.85 | 1010.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 973.35 | 976.37 | 0.00 | ORB-short ORB[975.00,981.00] vol=1.8x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:10:00 | 970.02 | 975.02 | 0.00 | T1 1.5R @ 970.02 |
| Target hit | 2025-07-18 12:35:00 | 968.35 | 966.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — BUY (started 2025-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 09:30:00 | 1018.00 | 1012.88 | 0.00 | ORB-long ORB[1003.45,1015.55] vol=1.9x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-08-01 09:35:00 | 1015.19 | 1013.09 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 11:10:00 | 1007.85 | 998.13 | 0.00 | ORB-long ORB[990.05,1005.05] vol=2.6x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-08-04 11:25:00 | 1004.56 | 998.55 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-23 09:30:00 | 966.20 | 2025-05-23 09:35:00 | 971.40 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-23 09:30:00 | 966.20 | 2025-05-23 10:25:00 | 972.40 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2025-05-27 10:20:00 | 994.75 | 2025-05-27 10:40:00 | 991.93 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-02 09:45:00 | 1006.65 | 2025-06-02 09:55:00 | 1013.40 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-06-02 09:45:00 | 1006.65 | 2025-06-02 10:00:00 | 1006.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-05 10:50:00 | 1050.50 | 2025-06-05 10:55:00 | 1053.67 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-10 10:50:00 | 1014.40 | 2025-06-10 11:25:00 | 1011.59 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-10 10:50:00 | 1014.40 | 2025-06-10 15:20:00 | 1010.00 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-11 10:35:00 | 1009.55 | 2025-06-11 12:00:00 | 1006.03 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-06-11 10:35:00 | 1009.55 | 2025-06-11 15:20:00 | 999.45 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2025-06-12 09:40:00 | 990.40 | 2025-06-12 09:50:00 | 993.18 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-17 09:40:00 | 965.00 | 2025-06-17 09:45:00 | 961.32 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-19 09:40:00 | 962.15 | 2025-06-19 09:45:00 | 966.56 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-06-19 09:40:00 | 962.15 | 2025-06-19 10:05:00 | 962.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-24 10:25:00 | 963.90 | 2025-06-24 10:30:00 | 961.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-25 09:50:00 | 977.05 | 2025-06-25 10:35:00 | 982.27 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-06-25 09:50:00 | 977.05 | 2025-06-25 10:40:00 | 977.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:30:00 | 976.45 | 2025-06-27 09:35:00 | 981.25 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-06-27 09:30:00 | 976.45 | 2025-06-27 09:40:00 | 976.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:00:00 | 961.70 | 2025-07-01 10:10:00 | 958.67 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-01 10:00:00 | 961.70 | 2025-07-01 10:40:00 | 961.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 11:10:00 | 968.50 | 2025-07-03 11:15:00 | 966.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-04 10:45:00 | 985.60 | 2025-07-04 11:00:00 | 982.68 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-08 11:10:00 | 965.00 | 2025-07-08 12:25:00 | 967.08 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-09 09:30:00 | 981.60 | 2025-07-09 09:40:00 | 986.13 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-07-09 09:30:00 | 981.60 | 2025-07-09 13:10:00 | 997.15 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2025-07-10 10:10:00 | 1006.70 | 2025-07-10 10:35:00 | 1011.53 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-10 10:10:00 | 1006.70 | 2025-07-10 12:20:00 | 1009.85 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-18 09:45:00 | 973.35 | 2025-07-18 10:10:00 | 970.02 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-18 09:45:00 | 973.35 | 2025-07-18 12:35:00 | 968.35 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-01 09:30:00 | 1018.00 | 2025-08-01 09:35:00 | 1015.19 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-04 11:10:00 | 1007.85 | 2025-08-04 11:25:00 | 1004.56 | STOP_HIT | 1.00 | -0.33% |
