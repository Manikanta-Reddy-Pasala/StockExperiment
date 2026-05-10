# Mahanagar Gas Ltd. (MGL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1173.50
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 3
- **Avg / median % per leg:** -0.13% / -0.29%
- **Sum % (uncompounded):** -1.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 0 | 11 | 3 | -0.12% | -1.6% |
| BUY @ 2nd Alert (retest1) | 14 | 3 | 21.4% | 0 | 11 | 3 | -0.12% | -1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.29% | -0.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.29% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 3 | 20.0% | 0 | 12 | 3 | -0.13% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:30:00 | 1120.80 | 1111.27 | 0.00 | ORB-long ORB[1105.00,1119.10] vol=3.2x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-02-18 10:40:00 | 1117.38 | 1111.91 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:15:00 | 1133.80 | 1129.37 | 0.00 | ORB-long ORB[1119.50,1130.90] vol=5.5x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:50:00 | 1138.44 | 1132.10 | 0.00 | T1 1.5R @ 1138.44 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 1133.80 | 1133.47 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 1201.20 | 1194.08 | 0.00 | ORB-long ORB[1188.00,1198.90] vol=1.6x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-02-25 09:40:00 | 1197.26 | 1194.54 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 1209.80 | 1202.99 | 0.00 | ORB-long ORB[1195.70,1204.40] vol=2.5x ATR=2.62 |
| Stop hit — per-position SL triggered | 2026-02-27 09:35:00 | 1207.18 | 1206.05 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:55:00 | 988.90 | 995.45 | 0.00 | ORB-short ORB[992.80,1006.20] vol=1.7x ATR=2.86 |
| Stop hit — per-position SL triggered | 2026-03-20 11:05:00 | 991.76 | 994.97 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 1153.25 | 1147.13 | 0.00 | ORB-long ORB[1132.80,1148.50] vol=1.5x ATR=3.40 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 1149.85 | 1148.72 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 1141.95 | 1131.98 | 0.00 | ORB-long ORB[1126.10,1139.75] vol=2.3x ATR=5.33 |
| Stop hit — per-position SL triggered | 2026-04-22 10:05:00 | 1136.62 | 1137.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 1145.00 | 1132.04 | 0.00 | ORB-long ORB[1127.00,1137.45] vol=1.7x ATR=5.33 |
| Stop hit — per-position SL triggered | 2026-04-27 13:30:00 | 1139.67 | 1141.82 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 1146.80 | 1142.34 | 0.00 | ORB-long ORB[1134.20,1146.40] vol=3.0x ATR=4.70 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 1142.10 | 1143.34 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 1137.80 | 1127.53 | 0.00 | ORB-long ORB[1121.10,1132.10] vol=1.7x ATR=4.52 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 1133.28 | 1129.13 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 1152.00 | 1146.15 | 0.00 | ORB-long ORB[1141.00,1149.50] vol=4.3x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:45:00 | 1156.73 | 1148.44 | 0.00 | T1 1.5R @ 1156.73 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1152.00 | 1149.85 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 1177.90 | 1173.70 | 0.00 | ORB-long ORB[1166.00,1177.20] vol=1.6x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:05:00 | 1182.99 | 1175.07 | 0.00 | T1 1.5R @ 1182.99 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1177.90 | 1175.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-18 10:30:00 | 1120.80 | 2026-02-18 10:40:00 | 1117.38 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-19 10:15:00 | 1133.80 | 2026-02-19 10:50:00 | 1138.44 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-19 10:15:00 | 1133.80 | 2026-02-19 11:00:00 | 1133.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:35:00 | 1201.20 | 2026-02-25 09:40:00 | 1197.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-27 09:30:00 | 1209.80 | 2026-02-27 09:35:00 | 1207.18 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-20 10:55:00 | 988.90 | 2026-03-20 11:05:00 | 991.76 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-21 10:05:00 | 1153.25 | 2026-04-21 10:40:00 | 1149.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-22 09:40:00 | 1141.95 | 2026-04-22 10:05:00 | 1136.62 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-27 09:40:00 | 1145.00 | 2026-04-27 13:30:00 | 1139.67 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-05-04 09:40:00 | 1146.80 | 2026-05-04 10:20:00 | 1142.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-05 09:40:00 | 1137.80 | 2026-05-05 09:50:00 | 1133.28 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1152.00 | 2026-05-06 09:45:00 | 1156.73 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1152.00 | 2026-05-06 10:15:00 | 1152.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 11:00:00 | 1177.90 | 2026-05-07 11:05:00 | 1182.99 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-05-07 11:00:00 | 1177.90 | 2026-05-07 11:15:00 | 1177.90 | STOP_HIT | 0.50 | 0.00% |
