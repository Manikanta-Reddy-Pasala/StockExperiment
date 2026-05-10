# Kalpataru Projects International Ltd. (KPIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1277.20
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 5
- **Avg / median % per leg:** 0.34% / 0.46%
- **Sum % (uncompounded):** 4.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.36% | 4.3% |
| BUY @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.36% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.23% | 0.5% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.23% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.34% | 4.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 1122.70 | 1111.13 | 0.00 | ORB-long ORB[1091.00,1102.00] vol=2.0x ATR=6.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:45:00 | 1132.54 | 1117.23 | 0.00 | T1 1.5R @ 1132.54 |
| Stop hit — per-position SL triggered | 2026-02-09 13:10:00 | 1122.70 | 1122.59 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 1116.00 | 1109.39 | 0.00 | ORB-long ORB[1092.70,1103.50] vol=4.3x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 1112.55 | 1110.65 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 1203.80 | 1193.52 | 0.00 | ORB-long ORB[1186.50,1196.90] vol=1.6x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:50:00 | 1210.68 | 1201.80 | 0.00 | T1 1.5R @ 1210.68 |
| Target hit | 2026-02-27 15:20:00 | 1232.30 | 1229.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1116.40 | 1108.26 | 0.00 | ORB-long ORB[1098.30,1108.00] vol=3.9x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-03-10 11:50:00 | 1113.15 | 1108.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 1173.20 | 1164.51 | 0.00 | ORB-long ORB[1150.90,1157.30] vol=4.0x ATR=5.24 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 1167.96 | 1164.74 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 1200.80 | 1195.73 | 0.00 | ORB-long ORB[1185.70,1199.00] vol=2.9x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:40:00 | 1207.86 | 1199.82 | 0.00 | T1 1.5R @ 1207.86 |
| Target hit | 2026-04-15 15:20:00 | 1210.50 | 1210.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:35:00 | 1250.00 | 1244.99 | 0.00 | ORB-long ORB[1238.40,1249.90] vol=2.0x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:40:00 | 1255.91 | 1247.42 | 0.00 | T1 1.5R @ 1255.91 |
| Stop hit — per-position SL triggered | 2026-04-21 10:45:00 | 1250.00 | 1247.42 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 1247.20 | 1257.18 | 0.00 | ORB-short ORB[1257.90,1272.90] vol=1.6x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:50:00 | 1241.47 | 1254.84 | 0.00 | T1 1.5R @ 1241.47 |
| Stop hit — per-position SL triggered | 2026-04-24 12:05:00 | 1247.20 | 1254.61 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 1260.00 | 1255.01 | 0.00 | ORB-long ORB[1243.70,1258.90] vol=1.8x ATR=4.69 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 1255.31 | 1255.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 1122.70 | 2026-02-09 10:45:00 | 1132.54 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2026-02-09 10:30:00 | 1122.70 | 2026-02-09 13:10:00 | 1122.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:00:00 | 1116.00 | 2026-02-17 11:30:00 | 1112.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-27 09:30:00 | 1203.80 | 2026-02-27 10:50:00 | 1210.68 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-27 09:30:00 | 1203.80 | 2026-02-27 15:20:00 | 1232.30 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2026-03-10 11:15:00 | 1116.40 | 2026-03-10 11:50:00 | 1113.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-10 10:00:00 | 1173.20 | 2026-04-10 10:05:00 | 1167.96 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-15 09:30:00 | 1200.80 | 2026-04-15 09:40:00 | 1207.86 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-15 09:30:00 | 1200.80 | 2026-04-15 15:20:00 | 1210.50 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2026-04-21 10:35:00 | 1250.00 | 2026-04-21 10:40:00 | 1255.91 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-21 10:35:00 | 1250.00 | 2026-04-21 10:45:00 | 1250.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 11:00:00 | 1247.20 | 2026-04-24 11:50:00 | 1241.47 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-24 11:00:00 | 1247.20 | 2026-04-24 12:05:00 | 1247.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:40:00 | 1260.00 | 2026-04-27 11:30:00 | 1255.31 | STOP_HIT | 1.00 | -0.37% |
