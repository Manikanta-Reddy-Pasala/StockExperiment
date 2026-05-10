# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1075.50
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 8
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.13% | 1.6% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.13% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 0 | 7 | 4 | 0.10% | 1.1% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 0 | 7 | 4 | 0.10% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 9 | 37.5% | 1 | 15 | 8 | 0.11% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 1051.55 | 1046.86 | 0.00 | ORB-long ORB[1038.40,1048.85] vol=1.6x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 1057.99 | 1049.12 | 0.00 | T1 1.5R @ 1057.99 |
| Stop hit — per-position SL triggered | 2026-02-10 10:00:00 | 1051.55 | 1050.86 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:55:00 | 1054.55 | 1061.45 | 0.00 | ORB-short ORB[1060.00,1074.80] vol=2.0x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:15:00 | 1049.85 | 1059.32 | 0.00 | T1 1.5R @ 1049.85 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 1054.55 | 1057.19 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 1038.85 | 1032.56 | 0.00 | ORB-long ORB[1025.05,1036.45] vol=3.5x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:45:00 | 1043.42 | 1035.88 | 0.00 | T1 1.5R @ 1043.42 |
| Stop hit — per-position SL triggered | 2026-02-17 13:05:00 | 1038.85 | 1040.69 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 1027.70 | 1036.07 | 0.00 | ORB-short ORB[1030.00,1045.45] vol=1.9x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:55:00 | 1021.48 | 1032.17 | 0.00 | T1 1.5R @ 1021.48 |
| Stop hit — per-position SL triggered | 2026-02-18 10:00:00 | 1027.70 | 1031.81 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 1030.55 | 1021.83 | 0.00 | ORB-long ORB[1016.00,1027.85] vol=1.9x ATR=3.08 |
| Target hit | 2026-02-20 15:20:00 | 1041.65 | 1031.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1010.00 | 1017.13 | 0.00 | ORB-short ORB[1016.00,1029.50] vol=1.8x ATR=4.01 |
| Stop hit — per-position SL triggered | 2026-02-24 09:50:00 | 1014.01 | 1016.06 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1037.40 | 1031.57 | 0.00 | ORB-long ORB[1028.15,1037.25] vol=2.7x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:25:00 | 1041.40 | 1032.99 | 0.00 | T1 1.5R @ 1041.40 |
| Stop hit — per-position SL triggered | 2026-02-25 11:35:00 | 1037.40 | 1033.42 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 1029.55 | 1032.58 | 0.00 | ORB-short ORB[1030.00,1037.40] vol=1.9x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-02-26 11:20:00 | 1031.21 | 1032.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 1023.60 | 1029.39 | 0.00 | ORB-short ORB[1027.00,1034.80] vol=4.7x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-02-27 11:20:00 | 1026.45 | 1029.27 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 1031.30 | 1026.00 | 0.00 | ORB-long ORB[1020.60,1031.00] vol=2.7x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 09:40:00 | 1037.63 | 1027.59 | 0.00 | T1 1.5R @ 1037.63 |
| Stop hit — per-position SL triggered | 2026-03-10 10:00:00 | 1031.30 | 1030.15 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:40:00 | 949.40 | 945.02 | 0.00 | ORB-long ORB[936.20,947.70] vol=4.3x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-04-06 09:50:00 | 945.46 | 945.98 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:55:00 | 943.00 | 935.82 | 0.00 | ORB-long ORB[927.05,937.60] vol=1.5x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-04-13 11:25:00 | 939.65 | 936.10 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 957.60 | 955.78 | 0.00 | ORB-long ORB[947.60,957.00] vol=2.5x ATR=2.60 |
| Stop hit — per-position SL triggered | 2026-04-16 09:35:00 | 955.00 | 955.79 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:30:00 | 929.80 | 933.73 | 0.00 | ORB-short ORB[931.20,940.80] vol=2.0x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 09:45:00 | 926.35 | 931.80 | 0.00 | T1 1.5R @ 926.35 |
| Stop hit — per-position SL triggered | 2026-04-20 09:55:00 | 929.80 | 931.56 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 1006.45 | 999.49 | 0.00 | ORB-long ORB[993.00,1003.95] vol=1.9x ATR=4.48 |
| Stop hit — per-position SL triggered | 2026-04-23 10:00:00 | 1001.97 | 1001.23 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 1065.50 | 1073.80 | 0.00 | ORB-short ORB[1065.60,1077.00] vol=5.5x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:35:00 | 1060.41 | 1072.67 | 0.00 | T1 1.5R @ 1060.41 |
| Stop hit — per-position SL triggered | 2026-05-06 11:55:00 | 1065.50 | 1072.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 1051.55 | 2026-02-10 09:45:00 | 1057.99 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-10 09:40:00 | 1051.55 | 2026-02-10 10:00:00 | 1051.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:55:00 | 1054.55 | 2026-02-13 11:15:00 | 1049.85 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-13 10:55:00 | 1054.55 | 2026-02-13 11:50:00 | 1054.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:15:00 | 1038.85 | 2026-02-17 11:45:00 | 1043.42 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-17 11:15:00 | 1038.85 | 2026-02-17 13:05:00 | 1038.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:50:00 | 1027.70 | 2026-02-18 09:55:00 | 1021.48 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-02-18 09:50:00 | 1027.70 | 2026-02-18 10:00:00 | 1027.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:50:00 | 1030.55 | 2026-02-20 15:20:00 | 1041.65 | TARGET_HIT | 1.00 | 1.08% |
| SELL | retest1 | 2026-02-24 09:45:00 | 1010.00 | 2026-02-24 09:50:00 | 1014.01 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-25 11:00:00 | 1037.40 | 2026-02-25 11:25:00 | 1041.40 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-25 11:00:00 | 1037.40 | 2026-02-25 11:35:00 | 1037.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 10:50:00 | 1029.55 | 2026-02-26 11:20:00 | 1031.21 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-02-27 11:10:00 | 1023.60 | 2026-02-27 11:20:00 | 1026.45 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-10 09:35:00 | 1031.30 | 2026-03-10 09:40:00 | 1037.63 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-03-10 09:35:00 | 1031.30 | 2026-03-10 10:00:00 | 1031.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-06 09:40:00 | 949.40 | 2026-04-06 09:50:00 | 945.46 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-13 10:55:00 | 943.00 | 2026-04-13 11:25:00 | 939.65 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-16 09:30:00 | 957.60 | 2026-04-16 09:35:00 | 955.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-20 09:30:00 | 929.80 | 2026-04-20 09:45:00 | 926.35 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-20 09:30:00 | 929.80 | 2026-04-20 09:55:00 | 929.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:45:00 | 1006.45 | 2026-04-23 10:00:00 | 1001.97 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-05-06 11:00:00 | 1065.50 | 2026-05-06 11:35:00 | 1060.41 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-05-06 11:00:00 | 1065.50 | 2026-05-06 11:55:00 | 1065.50 | STOP_HIT | 0.50 | 0.00% |
