# Global Health Ltd. (MEDANTA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1202.00
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
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 6
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 2.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.41% | 1.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.41% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 5 | 38.5% | 0 | 8 | 5 | 0.08% | 1.1% |
| SELL @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 0 | 8 | 5 | 0.08% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 7 | 41.2% | 1 | 10 | 6 | 0.16% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:20:00 | 1137.90 | 1138.28 | 0.00 | ORB-short ORB[1138.50,1151.00] vol=13.4x ATR=3.92 |
| Stop hit — per-position SL triggered | 2026-02-12 11:05:00 | 1141.82 | 1138.51 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:25:00 | 1158.30 | 1147.61 | 0.00 | ORB-long ORB[1131.20,1144.80] vol=2.7x ATR=3.47 |
| Stop hit — per-position SL triggered | 2026-02-23 10:45:00 | 1154.83 | 1148.68 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:50:00 | 1146.70 | 1147.93 | 0.00 | ORB-short ORB[1149.30,1162.90] vol=1.6x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:20:00 | 1142.01 | 1147.44 | 0.00 | T1 1.5R @ 1142.01 |
| Stop hit — per-position SL triggered | 2026-02-24 12:10:00 | 1146.70 | 1145.98 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 1145.60 | 1154.09 | 0.00 | ORB-short ORB[1149.10,1158.00] vol=1.8x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:45:00 | 1141.41 | 1151.57 | 0.00 | T1 1.5R @ 1141.41 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1145.60 | 1149.23 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:35:00 | 1106.30 | 1113.64 | 0.00 | ORB-short ORB[1110.40,1125.00] vol=1.7x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:20:00 | 1101.42 | 1110.75 | 0.00 | T1 1.5R @ 1101.42 |
| Stop hit — per-position SL triggered | 2026-03-05 12:05:00 | 1106.30 | 1105.36 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 1078.10 | 1086.34 | 0.00 | ORB-short ORB[1084.80,1097.20] vol=1.7x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:15:00 | 1072.82 | 1085.23 | 0.00 | T1 1.5R @ 1072.82 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 1078.10 | 1085.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 1029.20 | 1034.64 | 0.00 | ORB-short ORB[1031.00,1039.00] vol=2.3x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 12:00:00 | 1024.94 | 1032.23 | 0.00 | T1 1.5R @ 1024.94 |
| Stop hit — per-position SL triggered | 2026-03-19 12:50:00 | 1029.20 | 1031.53 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 1069.20 | 1068.47 | 0.00 | ORB-long ORB[1053.45,1067.00] vol=8.4x ATR=4.72 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 1064.48 | 1068.40 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 1081.80 | 1090.21 | 0.00 | ORB-short ORB[1089.65,1099.65] vol=1.6x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 1086.30 | 1086.76 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:10:00 | 1069.70 | 1078.57 | 0.00 | ORB-short ORB[1081.50,1095.00] vol=1.8x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-04-17 12:00:00 | 1072.53 | 1076.87 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:00:00 | 1092.75 | 1088.88 | 0.00 | ORB-long ORB[1078.60,1091.00] vol=1.6x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:20:00 | 1098.06 | 1090.81 | 0.00 | T1 1.5R @ 1098.06 |
| Target hit | 2026-04-22 15:20:00 | 1113.65 | 1103.68 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:20:00 | 1137.90 | 2026-02-12 11:05:00 | 1141.82 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-23 10:25:00 | 1158.30 | 2026-02-23 10:45:00 | 1154.83 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-24 10:50:00 | 1146.70 | 2026-02-24 11:20:00 | 1142.01 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-24 10:50:00 | 1146.70 | 2026-02-24 12:10:00 | 1146.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 10:30:00 | 1145.60 | 2026-02-25 10:45:00 | 1141.41 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-25 10:30:00 | 1145.60 | 2026-02-25 11:15:00 | 1145.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 09:35:00 | 1106.30 | 2026-03-05 10:20:00 | 1101.42 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-05 09:35:00 | 1106.30 | 2026-03-05 12:05:00 | 1106.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:35:00 | 1078.10 | 2026-03-13 11:15:00 | 1072.82 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-13 10:35:00 | 1078.10 | 2026-03-13 11:25:00 | 1078.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 11:00:00 | 1029.20 | 2026-03-19 12:00:00 | 1024.94 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-19 11:00:00 | 1029.20 | 2026-03-19 12:50:00 | 1029.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:40:00 | 1069.20 | 2026-04-10 09:55:00 | 1064.48 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-16 09:45:00 | 1081.80 | 2026-04-16 10:30:00 | 1086.30 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-17 11:10:00 | 1069.70 | 2026-04-17 12:00:00 | 1072.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-22 10:00:00 | 1092.75 | 2026-04-22 11:20:00 | 1098.06 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 10:00:00 | 1092.75 | 2026-04-22 15:20:00 | 1113.65 | TARGET_HIT | 0.50 | 1.91% |
