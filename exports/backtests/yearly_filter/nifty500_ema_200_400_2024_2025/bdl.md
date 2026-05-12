# Bharat Dynamics Ltd. (BDL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1447.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 16
- **Target hits / Stop hits / Partials:** 1 / 22 / 6
- **Avg / median % per leg:** -0.10% / -1.53%
- **Sum % (uncompounded):** -3.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 1 | 9 | 0 | -0.98% | -9.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 1 | 9 | 0 | -0.98% | -9.8% |
| SELL (all) | 19 | 12 | 63.2% | 0 | 13 | 6 | 0.36% | 6.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 12 | 63.2% | 0 | 13 | 6 | 0.36% | 6.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 13 | 44.8% | 1 | 22 | 6 | -0.10% | -3.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 1313.85 | 1389.36 | 1389.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 1312.00 | 1388.59 | 1388.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 10:15:00 | 1202.95 | 1194.12 | 1255.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 1202.95 | 1194.12 | 1255.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1116.45 | 1050.35 | 1115.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 1116.45 | 1050.35 | 1115.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 1127.40 | 1051.12 | 1116.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 1149.60 | 1051.12 | 1116.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1163.25 | 1052.24 | 1116.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 1164.15 | 1052.24 | 1116.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 13:15:00 | 1280.00 | 1152.75 | 1152.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 1340.35 | 1173.74 | 1167.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 1184.10 | 1201.30 | 1183.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 1184.10 | 1201.30 | 1183.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1184.10 | 1201.30 | 1183.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 1184.10 | 1201.30 | 1183.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1185.00 | 1201.13 | 1183.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:30:00 | 1182.20 | 1201.13 | 1183.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 1177.75 | 1200.90 | 1183.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:00:00 | 1177.75 | 1200.90 | 1183.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 1197.00 | 1200.86 | 1183.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 13:15:00 | 1197.95 | 1200.86 | 1183.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 09:15:00 | 1146.55 | 1200.31 | 1183.73 | SL hit (close<static) qty=1.00 sl=1177.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 1037.55 | 1183.80 | 1183.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 10:15:00 | 1029.80 | 1182.27 | 1183.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 1119.05 | 1092.87 | 1128.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 1119.05 | 1092.87 | 1128.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1119.05 | 1092.87 | 1128.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:00:00 | 1119.05 | 1092.87 | 1128.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1124.20 | 1093.18 | 1128.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:30:00 | 1124.60 | 1093.18 | 1128.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1132.15 | 1093.57 | 1128.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 10:00:00 | 1122.10 | 1095.42 | 1128.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 1113.70 | 1097.84 | 1129.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 11:00:00 | 1122.70 | 1098.30 | 1129.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:00:00 | 1118.00 | 1099.39 | 1128.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1130.80 | 1099.72 | 1127.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 1130.80 | 1099.72 | 1127.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1132.00 | 1100.04 | 1127.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 1137.45 | 1100.04 | 1127.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1116.75 | 1100.21 | 1127.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:45:00 | 1111.00 | 1100.29 | 1127.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:45:00 | 1109.40 | 1100.52 | 1127.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 1107.00 | 1101.52 | 1127.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 1163.45 | 1103.66 | 1127.10 | SL hit (close>static) qty=1.00 sl=1148.60 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 10:15:00 | 1307.05 | 1146.83 | 1146.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-25 13:15:00 | 1317.65 | 1151.74 | 1149.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1853.00 | 1853.14 | 1710.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1853.00 | 1853.14 | 1710.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1795.00 | 1883.62 | 1793.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1807.50 | 1883.62 | 1793.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 1790.90 | 1882.70 | 1793.19 | SL hit (close<static) qty=1.00 sl=1793.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 1590.40 | 1741.57 | 1741.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 1574.50 | 1739.91 | 1740.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 1570.40 | 1527.44 | 1593.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 1570.40 | 1527.44 | 1593.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1585.80 | 1529.84 | 1592.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 1588.40 | 1529.84 | 1592.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1608.40 | 1532.12 | 1592.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 1608.40 | 1532.12 | 1592.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1600.00 | 1532.79 | 1592.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1604.50 | 1532.79 | 1592.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1614.10 | 1534.40 | 1592.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 1619.50 | 1534.40 | 1592.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 1604.50 | 1536.42 | 1593.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 1604.50 | 1536.42 | 1593.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1583.80 | 1558.99 | 1596.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:45:00 | 1576.00 | 1559.20 | 1596.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 1580.70 | 1559.61 | 1596.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1579.00 | 1560.30 | 1596.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 1579.20 | 1560.49 | 1596.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1582.60 | 1561.08 | 1595.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 1599.00 | 1561.08 | 1595.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1598.00 | 1561.45 | 1595.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1598.00 | 1561.45 | 1595.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1584.60 | 1561.68 | 1595.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 1582.70 | 1561.87 | 1595.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1497.20 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1501.66 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1500.05 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1500.24 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1503.57 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1567.90 | 1549.23 | 1583.71 | SL hit (close>ema200) qty=0.50 sl=1549.23 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 1356.50 | 1333.43 | 1333.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 1373.00 | 1334.10 | 1333.67 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-01-27 13:15:00 | 1197.95 | 2025-01-28 09:15:00 | 1146.55 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2025-01-28 12:00:00 | 1216.60 | 2025-02-01 09:15:00 | 1338.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-04 09:15:00 | 1203.05 | 2025-02-10 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-02-04 11:45:00 | 1202.05 | 2025-02-10 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-02-05 09:15:00 | 1205.15 | 2025-02-10 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-02-07 10:00:00 | 1195.20 | 2025-02-10 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-02-07 10:45:00 | 1202.60 | 2025-02-10 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-02-07 14:15:00 | 1199.25 | 2025-02-10 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-02-10 11:45:00 | 1217.75 | 2025-02-10 13:15:00 | 1188.15 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-03-10 10:00:00 | 1122.10 | 2025-03-19 09:15:00 | 1163.45 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-03-11 09:15:00 | 1113.70 | 2025-03-19 09:15:00 | 1163.45 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2025-03-11 11:00:00 | 1122.70 | 2025-03-19 09:15:00 | 1163.45 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-03-12 10:00:00 | 1118.00 | 2025-03-19 09:15:00 | 1163.45 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-03-13 12:45:00 | 1111.00 | 2025-03-19 09:15:00 | 1163.45 | STOP_HIT | 1.00 | -4.72% |
| SELL | retest2 | 2025-03-13 14:45:00 | 1109.40 | 2025-03-19 09:15:00 | 1163.45 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2025-03-17 15:00:00 | 1107.00 | 2025-03-19 09:15:00 | 1163.45 | STOP_HIT | 1.00 | -5.10% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1807.50 | 2025-07-17 09:15:00 | 1790.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-23 11:45:00 | 1576.00 | 2025-09-26 14:15:00 | 1497.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 1580.70 | 2025-09-26 14:15:00 | 1501.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1579.00 | 2025-09-26 14:15:00 | 1500.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:00:00 | 1579.20 | 2025-09-26 14:15:00 | 1500.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-09-26 14:15:00 | 1503.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:45:00 | 1576.00 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2025-09-23 13:45:00 | 1580.70 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.81% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1579.00 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.70% |
| SELL | retest2 | 2025-09-24 10:00:00 | 1579.20 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1581.20 | 2025-11-24 09:15:00 | 1502.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1581.20 | 2025-12-01 09:15:00 | 1524.00 | STOP_HIT | 0.50 | 3.62% |
