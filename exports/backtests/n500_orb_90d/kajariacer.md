# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1105.00
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 2
- **Avg / median % per leg:** -0.11% / -0.34%
- **Sum % (uncompounded):** -1.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 2 | 6 | 2 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 2 | 6 | 2 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.36% | -1.4% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.36% | -1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 4 | 28.6% | 2 | 10 | 2 | -0.11% | -1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:35:00 | 921.40 | 929.13 | 0.00 | ORB-short ORB[926.00,938.35] vol=3.9x ATR=3.64 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 925.04 | 926.10 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 950.10 | 940.32 | 0.00 | ORB-long ORB[924.00,935.95] vol=1.8x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 946.69 | 940.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 976.15 | 985.97 | 0.00 | ORB-short ORB[981.10,992.20] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-02-24 11:10:00 | 979.29 | 985.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 966.15 | 959.77 | 0.00 | ORB-long ORB[952.65,957.25] vol=6.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 962.82 | 961.00 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 924.20 | 931.19 | 0.00 | ORB-short ORB[932.95,942.30] vol=2.5x ATR=3.88 |
| Stop hit — per-position SL triggered | 2026-03-10 10:20:00 | 928.08 | 926.28 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:05:00 | 927.00 | 933.60 | 0.00 | ORB-short ORB[932.85,943.00] vol=2.1x ATR=2.64 |
| Stop hit — per-position SL triggered | 2026-03-19 11:10:00 | 929.64 | 933.47 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:45:00 | 998.10 | 986.28 | 0.00 | ORB-long ORB[973.80,988.50] vol=3.9x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 994.65 | 986.56 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1112.00 | 1104.12 | 0.00 | ORB-long ORB[1093.30,1107.70] vol=3.0x ATR=4.09 |
| Stop hit — per-position SL triggered | 2026-04-10 13:40:00 | 1107.91 | 1106.43 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1150.40 | 1142.55 | 0.00 | ORB-long ORB[1135.45,1146.00] vol=1.7x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:45:00 | 1158.88 | 1145.54 | 0.00 | T1 1.5R @ 1158.88 |
| Target hit | 2026-04-15 13:35:00 | 1158.45 | 1159.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1200.65 | 1190.46 | 0.00 | ORB-long ORB[1180.10,1194.00] vol=1.8x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:50:00 | 1206.48 | 1193.40 | 0.00 | T1 1.5R @ 1206.48 |
| Target hit | 2026-04-22 10:25:00 | 1202.40 | 1203.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 1232.30 | 1224.70 | 0.00 | ORB-long ORB[1213.00,1230.10] vol=1.9x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-04-27 10:10:00 | 1228.08 | 1227.24 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 1121.00 | 1113.10 | 0.00 | ORB-long ORB[1105.00,1120.00] vol=3.1x ATR=4.03 |
| Stop hit — per-position SL triggered | 2026-05-07 10:55:00 | 1116.97 | 1114.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-16 09:35:00 | 921.40 | 2026-02-16 09:40:00 | 925.04 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-17 10:25:00 | 950.10 | 2026-02-17 10:30:00 | 946.69 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-24 11:00:00 | 976.15 | 2026-02-24 11:10:00 | 979.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 11:05:00 | 966.15 | 2026-02-26 11:30:00 | 962.82 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-10 09:35:00 | 924.20 | 2026-03-10 10:20:00 | 928.08 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-19 11:05:00 | 927.00 | 2026-03-19 11:10:00 | 929.64 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-07 10:45:00 | 998.10 | 2026-04-07 10:50:00 | 994.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-10 11:00:00 | 1112.00 | 2026-04-10 13:40:00 | 1107.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1150.40 | 2026-04-15 09:45:00 | 1158.88 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1150.40 | 2026-04-15 13:35:00 | 1158.45 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1200.65 | 2026-04-22 09:50:00 | 1206.48 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1200.65 | 2026-04-22 10:25:00 | 1202.40 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1232.30 | 2026-04-27 10:10:00 | 1228.08 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-07 10:40:00 | 1121.00 | 2026-05-07 10:55:00 | 1116.97 | STOP_HIT | 1.00 | -0.36% |
