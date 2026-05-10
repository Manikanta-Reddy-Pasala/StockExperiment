# Zen Technologies Ltd. (ZENTEC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1626.00
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 3
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 0.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | 0.03% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.19% | 0.6% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.19% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.07% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 1334.40 | 1342.13 | 0.00 | ORB-short ORB[1338.30,1351.60] vol=1.8x ATR=5.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 1326.87 | 1336.16 | 0.00 | T1 1.5R @ 1326.87 |
| Target hit | 2026-02-10 14:40:00 | 1330.50 | 1328.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 1351.80 | 1340.21 | 0.00 | ORB-long ORB[1322.60,1338.90] vol=1.7x ATR=6.07 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 1345.73 | 1345.52 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 1305.20 | 1317.85 | 0.00 | ORB-short ORB[1321.00,1333.40] vol=1.6x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-02-23 10:50:00 | 1308.98 | 1316.83 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 1332.00 | 1325.62 | 0.00 | ORB-long ORB[1324.20,1331.20] vol=2.9x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-02-25 11:55:00 | 1328.59 | 1327.21 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:10:00 | 1363.60 | 1353.29 | 0.00 | ORB-long ORB[1337.00,1355.20] vol=2.1x ATR=4.88 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 1358.72 | 1355.23 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:45:00 | 1443.30 | 1437.02 | 0.00 | ORB-long ORB[1420.30,1440.80] vol=1.8x ATR=6.97 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 1436.33 | 1437.47 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:40:00 | 1462.20 | 1443.39 | 0.00 | ORB-long ORB[1424.10,1443.50] vol=4.1x ATR=6.34 |
| Stop hit — per-position SL triggered | 2026-03-19 10:45:00 | 1455.86 | 1443.93 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 11:05:00 | 1370.00 | 1362.14 | 0.00 | ORB-long ORB[1346.20,1366.30] vol=1.7x ATR=6.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 11:30:00 | 1379.07 | 1363.42 | 0.00 | T1 1.5R @ 1379.07 |
| Stop hit — per-position SL triggered | 2026-04-06 12:35:00 | 1370.00 | 1365.82 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1793.60 | 1745.53 | 0.00 | ORB-long ORB[1648.80,1674.90] vol=5.9x ATR=18.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 1821.96 | 1760.82 | 0.00 | T1 1.5R @ 1821.96 |
| Stop hit — per-position SL triggered | 2026-04-21 09:55:00 | 1793.60 | 1771.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:35:00 | 1334.40 | 2026-02-10 09:40:00 | 1326.87 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-10 09:35:00 | 1334.40 | 2026-02-10 14:40:00 | 1330.50 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-17 09:45:00 | 1351.80 | 2026-02-17 10:45:00 | 1345.73 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-02-23 10:40:00 | 1305.20 | 2026-02-23 10:50:00 | 1308.98 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-25 10:45:00 | 1332.00 | 2026-02-25 11:55:00 | 1328.59 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-26 10:10:00 | 1363.60 | 2026-02-26 10:20:00 | 1358.72 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-18 09:45:00 | 1443.30 | 2026-03-18 09:55:00 | 1436.33 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-19 10:40:00 | 1462.20 | 2026-03-19 10:45:00 | 1455.86 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-06 11:05:00 | 1370.00 | 2026-04-06 11:30:00 | 1379.07 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-06 11:05:00 | 1370.00 | 2026-04-06 12:35:00 | 1370.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1793.60 | 2026-04-21 09:45:00 | 1821.96 | PARTIAL | 0.50 | 1.58% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1793.60 | 2026-04-21 09:55:00 | 1793.60 | STOP_HIT | 0.50 | 0.00% |
