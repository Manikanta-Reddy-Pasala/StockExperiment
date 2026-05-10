# Home First Finance Company India Ltd. (HOMEFIRST)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1200.60
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 1.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.03% | 0.2% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.03% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.38% | 1.5% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.38% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.17% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 1176.70 | 1185.83 | 0.00 | ORB-short ORB[1183.20,1194.10] vol=1.9x ATR=4.03 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 1180.73 | 1181.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 1151.60 | 1146.63 | 0.00 | ORB-long ORB[1135.30,1150.00] vol=2.5x ATR=4.79 |
| Stop hit — per-position SL triggered | 2026-02-20 11:50:00 | 1146.81 | 1147.27 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:10:00 | 1065.00 | 1075.08 | 0.00 | ORB-short ORB[1071.00,1087.00] vol=1.8x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:55:00 | 1057.79 | 1070.41 | 0.00 | T1 1.5R @ 1057.79 |
| Target hit | 2026-03-05 15:20:00 | 1048.50 | 1047.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 1118.75 | 1112.31 | 0.00 | ORB-long ORB[1103.30,1109.95] vol=2.6x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:05:00 | 1124.64 | 1117.53 | 0.00 | T1 1.5R @ 1124.64 |
| Stop hit — per-position SL triggered | 2026-04-17 11:30:00 | 1118.75 | 1122.53 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 1146.60 | 1139.18 | 0.00 | ORB-long ORB[1134.00,1141.30] vol=1.9x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:55:00 | 1152.48 | 1140.78 | 0.00 | T1 1.5R @ 1152.48 |
| Stop hit — per-position SL triggered | 2026-04-27 10:25:00 | 1146.60 | 1142.32 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1150.90 | 1135.24 | 0.00 | ORB-long ORB[1125.00,1139.95] vol=3.7x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 1146.14 | 1138.85 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-05-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:40:00 | 1197.60 | 1199.38 | 0.00 | ORB-short ORB[1200.10,1215.60] vol=2.0x ATR=4.46 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 1202.06 | 1199.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:50:00 | 1176.70 | 2026-02-18 10:50:00 | 1180.73 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-20 10:50:00 | 1151.60 | 2026-02-20 11:50:00 | 1146.81 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-05 11:10:00 | 1065.00 | 2026-03-05 11:55:00 | 1057.79 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-03-05 11:10:00 | 1065.00 | 2026-03-05 15:20:00 | 1048.50 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2026-04-17 09:35:00 | 1118.75 | 2026-04-17 10:05:00 | 1124.64 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-17 09:35:00 | 1118.75 | 2026-04-17 11:30:00 | 1118.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1146.60 | 2026-04-27 09:55:00 | 1152.48 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1146.60 | 2026-04-27 10:25:00 | 1146.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1150.90 | 2026-04-29 10:50:00 | 1146.14 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-05-08 10:40:00 | 1197.60 | 2026-05-08 11:00:00 | 1202.06 | STOP_HIT | 1.00 | -0.37% |
