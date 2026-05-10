# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1342.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 7
- **Target hits / Stop hits / Partials:** 4 / 7 / 6
- **Avg / median % per leg:** 0.70% / 0.45%
- **Sum % (uncompounded):** 11.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.78% | 7.8% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.78% | 7.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.59% | 4.1% |
| SELL @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.59% | 4.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 10 | 58.8% | 4 | 7 | 6 | 0.70% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 1058.90 | 1063.13 | 0.00 | ORB-short ORB[1060.00,1073.30] vol=1.7x ATR=3.97 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 1062.87 | 1062.52 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 1032.10 | 1026.63 | 0.00 | ORB-long ORB[1016.90,1028.20] vol=1.8x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:50:00 | 1037.48 | 1029.66 | 0.00 | T1 1.5R @ 1037.48 |
| Target hit | 2026-02-17 13:55:00 | 1036.60 | 1037.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 1050.00 | 1045.29 | 0.00 | ORB-long ORB[1035.70,1049.70] vol=1.9x ATR=4.77 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 1045.23 | 1045.41 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:45:00 | 1022.10 | 1026.18 | 0.00 | ORB-short ORB[1023.80,1034.40] vol=2.4x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 1016.58 | 1023.61 | 0.00 | T1 1.5R @ 1016.58 |
| Target hit | 2026-02-19 15:20:00 | 1002.50 | 1016.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:50:00 | 1080.50 | 1072.07 | 0.00 | ORB-long ORB[1063.45,1073.90] vol=1.9x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:00:00 | 1086.91 | 1074.71 | 0.00 | T1 1.5R @ 1086.91 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 1080.50 | 1075.16 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 1115.25 | 1099.59 | 0.00 | ORB-long ORB[1090.00,1104.95] vol=2.4x ATR=5.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:00:00 | 1124.17 | 1106.82 | 0.00 | T1 1.5R @ 1124.17 |
| Target hit | 2026-04-15 15:20:00 | 1194.75 | 1197.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 1244.85 | 1238.67 | 0.00 | ORB-long ORB[1225.20,1243.80] vol=2.1x ATR=5.11 |
| Stop hit — per-position SL triggered | 2026-04-21 09:55:00 | 1239.74 | 1239.00 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 1232.20 | 1236.22 | 0.00 | ORB-short ORB[1235.50,1246.00] vol=2.1x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 1226.64 | 1236.00 | 0.00 | T1 1.5R @ 1226.64 |
| Target hit | 2026-04-28 15:20:00 | 1219.05 | 1229.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 1244.00 | 1234.09 | 0.00 | ORB-long ORB[1221.60,1236.45] vol=5.3x ATR=5.59 |
| Stop hit — per-position SL triggered | 2026-04-29 10:40:00 | 1238.41 | 1236.65 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 1271.30 | 1274.09 | 0.00 | ORB-short ORB[1272.00,1284.00] vol=3.2x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:35:00 | 1264.63 | 1273.00 | 0.00 | T1 1.5R @ 1264.63 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1271.30 | 1270.95 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 1305.80 | 1290.37 | 0.00 | ORB-long ORB[1285.40,1301.80] vol=3.4x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-05-07 10:50:00 | 1301.30 | 1292.70 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:40:00 | 1058.90 | 2026-02-11 09:45:00 | 1062.87 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-17 09:40:00 | 1032.10 | 2026-02-17 09:50:00 | 1037.48 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-17 09:40:00 | 1032.10 | 2026-02-17 13:55:00 | 1036.60 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-18 10:45:00 | 1050.00 | 2026-02-18 11:00:00 | 1045.23 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-02-19 09:45:00 | 1022.10 | 2026-02-19 11:15:00 | 1016.58 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-19 09:45:00 | 1022.10 | 2026-02-19 15:20:00 | 1002.50 | TARGET_HIT | 0.50 | 1.92% |
| BUY | retest1 | 2026-04-10 09:50:00 | 1080.50 | 2026-04-10 10:00:00 | 1086.91 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-10 09:50:00 | 1080.50 | 2026-04-10 10:05:00 | 1080.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:45:00 | 1115.25 | 2026-04-15 10:00:00 | 1124.17 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2026-04-15 09:45:00 | 1115.25 | 2026-04-15 15:20:00 | 1194.75 | TARGET_HIT | 0.50 | 7.13% |
| BUY | retest1 | 2026-04-21 09:50:00 | 1244.85 | 2026-04-21 09:55:00 | 1239.74 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-28 11:10:00 | 1232.20 | 2026-04-28 11:25:00 | 1226.64 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-28 11:10:00 | 1232.20 | 2026-04-28 15:20:00 | 1219.05 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2026-04-29 10:10:00 | 1244.00 | 2026-04-29 10:40:00 | 1238.41 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-05-06 09:50:00 | 1271.30 | 2026-05-06 10:35:00 | 1264.63 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-06 09:50:00 | 1271.30 | 2026-05-06 11:15:00 | 1271.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:45:00 | 1305.80 | 2026-05-07 10:50:00 | 1301.30 | STOP_HIT | 1.00 | -0.34% |
