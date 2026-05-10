# UTI Asset Management Company Ltd. (UTIAMC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 973.15
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 6
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.10% | 1.3% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.10% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.05% | 0.5% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.05% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 9 | 39.1% | 3 | 14 | 6 | 0.08% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:55:00 | 1081.40 | 1076.74 | 0.00 | ORB-long ORB[1061.25,1072.75] vol=3.6x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:35:00 | 1086.61 | 1079.01 | 0.00 | T1 1.5R @ 1086.61 |
| Target hit | 2026-02-10 13:40:00 | 1087.65 | 1088.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 1066.45 | 1060.73 | 0.00 | ORB-long ORB[1056.00,1064.75] vol=2.7x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-02-13 11:20:00 | 1063.15 | 1061.28 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 1057.65 | 1062.84 | 0.00 | ORB-short ORB[1061.25,1075.00] vol=5.0x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 1052.67 | 1060.57 | 0.00 | T1 1.5R @ 1052.67 |
| Target hit | 2026-02-24 15:20:00 | 1052.70 | 1051.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 1061.00 | 1059.48 | 0.00 | ORB-long ORB[1049.75,1059.20] vol=2.6x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:10:00 | 1065.58 | 1059.76 | 0.00 | T1 1.5R @ 1065.58 |
| Stop hit — per-position SL triggered | 2026-02-25 10:35:00 | 1061.00 | 1060.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 971.85 | 981.13 | 0.00 | ORB-short ORB[981.60,987.65] vol=2.2x ATR=3.52 |
| Stop hit — per-position SL triggered | 2026-03-06 13:00:00 | 975.37 | 976.25 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 967.00 | 960.04 | 0.00 | ORB-long ORB[956.80,963.95] vol=3.4x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-03-10 11:00:00 | 963.58 | 961.37 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 964.85 | 969.96 | 0.00 | ORB-short ORB[965.35,976.40] vol=2.4x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:40:00 | 959.48 | 966.75 | 0.00 | T1 1.5R @ 959.48 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 964.85 | 966.41 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:30:00 | 919.40 | 922.35 | 0.00 | ORB-short ORB[920.60,933.70] vol=2.5x ATR=3.67 |
| Stop hit — per-position SL triggered | 2026-03-23 09:40:00 | 923.07 | 922.18 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:05:00 | 960.25 | 953.89 | 0.00 | ORB-long ORB[945.05,957.40] vol=5.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 957.70 | 954.12 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 1017.05 | 1013.73 | 0.00 | ORB-long ORB[1000.00,1015.00] vol=1.8x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:55:00 | 1021.69 | 1015.57 | 0.00 | T1 1.5R @ 1021.69 |
| Target hit | 2026-04-17 15:20:00 | 1036.30 | 1029.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1079.75 | 1073.79 | 0.00 | ORB-long ORB[1068.00,1078.50] vol=1.6x ATR=4.12 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 1075.63 | 1075.85 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 966.45 | 945.62 | 0.00 | ORB-long ORB[931.00,943.80] vol=1.7x ATR=6.94 |
| Stop hit — per-position SL triggered | 2026-04-27 10:35:00 | 959.51 | 950.76 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 955.15 | 959.10 | 0.00 | ORB-short ORB[960.50,968.90] vol=3.1x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 956.90 | 959.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 964.30 | 958.13 | 0.00 | ORB-long ORB[947.70,962.05] vol=1.8x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-05-04 09:40:00 | 961.37 | 958.52 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:35:00 | 959.90 | 964.53 | 0.00 | ORB-short ORB[962.05,970.00] vol=1.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:45:00 | 956.19 | 963.93 | 0.00 | T1 1.5R @ 956.19 |
| Stop hit — per-position SL triggered | 2026-05-06 12:10:00 | 959.90 | 962.38 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 963.65 | 968.45 | 0.00 | ORB-short ORB[967.45,976.05] vol=1.6x ATR=3.74 |
| Stop hit — per-position SL triggered | 2026-05-07 10:30:00 | 967.39 | 964.61 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 979.20 | 976.08 | 0.00 | ORB-long ORB[971.20,978.90] vol=2.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-05-08 11:30:00 | 976.95 | 976.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:55:00 | 1081.40 | 2026-02-10 10:35:00 | 1086.61 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-10 09:55:00 | 1081.40 | 2026-02-10 13:40:00 | 1087.65 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-13 11:00:00 | 1066.45 | 2026-02-13 11:20:00 | 1063.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1057.65 | 2026-02-24 11:45:00 | 1052.67 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1057.65 | 2026-02-24 15:20:00 | 1052.70 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-25 09:50:00 | 1061.00 | 2026-02-25 10:10:00 | 1065.58 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-25 09:50:00 | 1061.00 | 2026-02-25 10:35:00 | 1061.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 971.85 | 2026-03-06 13:00:00 | 975.37 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-10 10:40:00 | 967.00 | 2026-03-10 11:00:00 | 963.58 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 09:45:00 | 964.85 | 2026-03-13 10:40:00 | 959.48 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-13 09:45:00 | 964.85 | 2026-03-13 10:50:00 | 964.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-23 09:30:00 | 919.40 | 2026-03-23 09:40:00 | 923.07 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-10 11:05:00 | 960.25 | 2026-04-10 11:15:00 | 957.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-17 09:50:00 | 1017.05 | 2026-04-17 09:55:00 | 1021.69 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-17 09:50:00 | 1017.05 | 2026-04-17 15:20:00 | 1036.30 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1079.75 | 2026-04-22 09:40:00 | 1075.63 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-27 10:10:00 | 966.45 | 2026-04-27 10:35:00 | 959.51 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2026-04-29 11:10:00 | 955.15 | 2026-04-29 11:20:00 | 956.90 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-05-04 09:35:00 | 964.30 | 2026-05-04 09:40:00 | 961.37 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-06 10:35:00 | 959.90 | 2026-05-06 10:45:00 | 956.19 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-06 10:35:00 | 959.90 | 2026-05-06 12:10:00 | 959.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 09:35:00 | 963.65 | 2026-05-07 10:30:00 | 967.39 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-08 11:15:00 | 979.20 | 2026-05-08 11:30:00 | 976.95 | STOP_HIT | 1.00 | -0.23% |
