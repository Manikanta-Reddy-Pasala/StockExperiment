# Lloyds Metals And Energy Ltd. (LLOYDSME)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1738.70
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 4
- **Avg / median % per leg:** 0.07% / -0.22%
- **Sum % (uncompounded):** 1.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 2 | 7 | 2 | 0.05% | 0.5% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 2 | 7 | 2 | 0.05% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.11% | 0.7% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.11% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 3 | 11 | 4 | 0.07% | 1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:10:00 | 1233.00 | 1239.42 | 0.00 | ORB-short ORB[1236.60,1252.00] vol=1.8x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:10:00 | 1226.03 | 1236.94 | 0.00 | T1 1.5R @ 1226.03 |
| Stop hit — per-position SL triggered | 2026-02-11 12:30:00 | 1233.00 | 1232.01 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:10:00 | 1242.10 | 1225.72 | 0.00 | ORB-long ORB[1208.10,1220.00] vol=4.1x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-02-12 10:35:00 | 1236.70 | 1232.49 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 1186.80 | 1182.63 | 0.00 | ORB-long ORB[1173.00,1184.50] vol=3.0x ATR=3.51 |
| Stop hit — per-position SL triggered | 2026-02-16 11:35:00 | 1183.29 | 1182.97 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 1194.40 | 1189.36 | 0.00 | ORB-long ORB[1182.60,1190.00] vol=1.6x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:25:00 | 1199.39 | 1193.45 | 0.00 | T1 1.5R @ 1199.39 |
| Target hit | 2026-02-17 10:40:00 | 1201.20 | 1201.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 1183.40 | 1203.14 | 0.00 | ORB-short ORB[1191.70,1205.40] vol=2.4x ATR=5.82 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 1189.22 | 1202.50 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:50:00 | 1184.00 | 1164.97 | 0.00 | ORB-long ORB[1151.20,1164.90] vol=1.5x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:05:00 | 1193.53 | 1174.92 | 0.00 | T1 1.5R @ 1193.53 |
| Target hit | 2026-03-17 11:15:00 | 1197.30 | 1200.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-03-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:55:00 | 1200.40 | 1214.13 | 0.00 | ORB-short ORB[1212.00,1228.10] vol=3.4x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 12:30:00 | 1190.34 | 1201.98 | 0.00 | T1 1.5R @ 1190.34 |
| Target hit | 2026-03-19 15:20:00 | 1191.70 | 1193.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 1532.00 | 1522.35 | 0.00 | ORB-long ORB[1510.10,1526.90] vol=4.7x ATR=5.29 |
| Stop hit — per-position SL triggered | 2026-04-16 09:40:00 | 1526.71 | 1522.67 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:50:00 | 1645.10 | 1635.04 | 0.00 | ORB-long ORB[1625.10,1641.00] vol=2.1x ATR=6.15 |
| Stop hit — per-position SL triggered | 2026-04-20 12:55:00 | 1638.95 | 1639.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1668.00 | 1661.33 | 0.00 | ORB-long ORB[1645.00,1663.00] vol=5.7x ATR=3.64 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 1664.36 | 1663.20 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 1789.00 | 1783.79 | 0.00 | ORB-long ORB[1771.80,1787.00] vol=2.9x ATR=5.90 |
| Stop hit — per-position SL triggered | 2026-05-04 12:30:00 | 1783.10 | 1788.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 1824.00 | 1813.10 | 0.00 | ORB-long ORB[1795.00,1817.00] vol=5.4x ATR=7.23 |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 1816.77 | 1817.09 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 1745.60 | 1758.51 | 0.00 | ORB-short ORB[1750.00,1773.00] vol=1.6x ATR=8.74 |
| Stop hit — per-position SL triggered | 2026-05-07 10:10:00 | 1754.34 | 1750.70 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 1708.00 | 1716.18 | 0.00 | ORB-short ORB[1710.00,1730.00] vol=2.2x ATR=6.64 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 1714.64 | 1717.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 10:10:00 | 1233.00 | 2026-02-11 11:10:00 | 1226.03 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-11 10:10:00 | 1233.00 | 2026-02-11 12:30:00 | 1233.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:10:00 | 1242.10 | 2026-02-12 10:35:00 | 1236.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-16 11:05:00 | 1186.80 | 2026-02-16 11:35:00 | 1183.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-17 09:40:00 | 1194.40 | 2026-02-17 10:25:00 | 1199.39 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-17 09:40:00 | 1194.40 | 2026-02-17 10:40:00 | 1201.20 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-06 10:55:00 | 1183.40 | 2026-03-06 11:00:00 | 1189.22 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-03-17 09:50:00 | 1184.00 | 2026-03-17 10:05:00 | 1193.53 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2026-03-17 09:50:00 | 1184.00 | 2026-03-17 11:15:00 | 1197.30 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2026-03-19 09:55:00 | 1200.40 | 2026-03-19 12:30:00 | 1190.34 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2026-03-19 09:55:00 | 1200.40 | 2026-03-19 15:20:00 | 1191.70 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-16 09:35:00 | 1532.00 | 2026-04-16 09:40:00 | 1526.71 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-20 10:50:00 | 1645.10 | 2026-04-20 12:55:00 | 1638.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1668.00 | 2026-04-22 09:45:00 | 1664.36 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-04 09:45:00 | 1789.00 | 2026-05-04 12:30:00 | 1783.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-05 09:50:00 | 1824.00 | 2026-05-05 10:15:00 | 1816.77 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-05-07 09:35:00 | 1745.60 | 2026-05-07 10:10:00 | 1754.34 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-08 09:50:00 | 1708.00 | 2026-05-08 09:55:00 | 1714.64 | STOP_HIT | 1.00 | -0.39% |
