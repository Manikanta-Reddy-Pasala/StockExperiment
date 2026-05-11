# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1655.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 17 |
| TARGET_HIT | 18 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 25
- **Target hits / Stop hits / Partials:** 18 / 26 / 17
- **Avg / median % per leg:** 3.48% / 4.84%
- **Sum % (uncompounded):** 212.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 2 | 7 | 0 | 0.67% | 6.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 2 | 7 | 0 | 0.67% | 6.0% |
| SELL (all) | 52 | 34 | 65.4% | 16 | 19 | 17 | 3.97% | 206.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 34 | 65.4% | 16 | 19 | 17 | 3.97% | 206.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 61 | 36 | 59.0% | 18 | 26 | 17 | 3.48% | 212.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 14:15:00 | 1226.70 | 1270.14 | 1270.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 14:15:00 | 1207.35 | 1263.85 | 1266.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 12:15:00 | 1251.18 | 1249.26 | 1258.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-30 12:30:00 | 1250.50 | 1249.26 | 1258.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 1254.40 | 1249.05 | 1258.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:45:00 | 1252.35 | 1249.05 | 1258.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 1260.03 | 1249.14 | 1258.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 13:00:00 | 1260.03 | 1249.14 | 1258.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 1257.97 | 1249.23 | 1258.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 15:15:00 | 1254.50 | 1249.32 | 1258.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 10:00:00 | 1250.72 | 1249.38 | 1258.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 15:00:00 | 1251.05 | 1249.45 | 1257.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 1241.45 | 1249.52 | 1257.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1245.80 | 1249.49 | 1257.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 11:00:00 | 1233.58 | 1249.33 | 1257.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 1239.00 | 1246.26 | 1255.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:30:00 | 1237.00 | 1246.09 | 1255.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 10:00:00 | 1237.28 | 1246.01 | 1254.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1236.80 | 1242.64 | 1252.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:30:00 | 1244.28 | 1242.64 | 1252.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 1245.95 | 1239.05 | 1249.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:15:00 | 1254.00 | 1239.05 | 1249.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 1255.00 | 1239.21 | 1249.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 13:45:00 | 1243.08 | 1239.51 | 1249.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 09:15:00 | 1261.20 | 1239.63 | 1249.21 | SL hit (close>static) qty=1.00 sl=1260.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1249.68 | 1163.36 | 1163.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 1278.43 | 1175.93 | 1169.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 13:15:00 | 1649.98 | 1654.20 | 1580.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 13:45:00 | 1651.15 | 1654.20 | 1580.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1650.63 | 1690.32 | 1640.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 15:15:00 | 1665.90 | 1689.18 | 1640.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 11:30:00 | 1665.63 | 1685.56 | 1642.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 12:00:00 | 1667.30 | 1685.56 | 1642.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 13:45:00 | 1666.15 | 1685.11 | 1642.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1650.98 | 1684.15 | 1643.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-23 14:15:00 | 1629.58 | 1682.17 | 1643.10 | SL hit (close<static) qty=1.00 sl=1639.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 13:15:00 | 1551.48 | 1614.88 | 1614.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 14:15:00 | 1547.45 | 1614.20 | 1614.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 1540.05 | 1535.43 | 1562.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 15:00:00 | 1540.05 | 1535.43 | 1562.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1564.85 | 1535.93 | 1561.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:45:00 | 1564.30 | 1535.93 | 1561.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 1565.60 | 1536.22 | 1561.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:00:00 | 1565.60 | 1536.22 | 1561.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1567.00 | 1536.53 | 1561.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:45:00 | 1565.15 | 1536.53 | 1561.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 1515.23 | 1507.97 | 1532.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:30:00 | 1516.95 | 1507.97 | 1532.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1518.68 | 1507.86 | 1531.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 1534.88 | 1507.86 | 1531.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1556.30 | 1508.34 | 1531.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1556.30 | 1508.34 | 1531.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1556.63 | 1508.82 | 1532.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 14:15:00 | 1545.00 | 1509.89 | 1532.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1537.53 | 1510.72 | 1532.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:15:00 | 1467.75 | 1508.86 | 1529.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 11:15:00 | 1460.65 | 1508.37 | 1529.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-15 09:15:00 | 1390.50 | 1493.75 | 1520.07 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 1391.55 | 1335.17 | 1334.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 1395.65 | 1336.34 | 1335.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 1391.20 | 1391.93 | 1372.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 1391.20 | 1391.93 | 1372.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1405.20 | 1430.26 | 1404.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:45:00 | 1408.75 | 1429.47 | 1404.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 1409.40 | 1429.47 | 1404.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 1391.35 | 1428.88 | 1404.34 | SL hit (close<static) qty=1.00 sl=1402.10 alert=retest2 |

### Cycle 5 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1543.00 | 1577.25 | 1577.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1536.80 | 1572.93 | 1575.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1616.00 | 1572.87 | 1574.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1616.00 | 1572.87 | 1574.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1616.00 | 1572.87 | 1574.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 1573.90 | 1576.27 | 1576.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:30:00 | 1575.25 | 1576.23 | 1576.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:45:00 | 1575.00 | 1576.16 | 1576.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1564.55 | 1576.22 | 1576.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1560.30 | 1576.06 | 1576.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 1551.55 | 1575.82 | 1576.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 1546.30 | 1575.34 | 1576.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 1556.55 | 1571.96 | 1574.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1585.40 | 1566.49 | 1571.17 | SL hit (close>static) qty=1.00 sl=1584.40 alert=retest2 |

### Cycle 6 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1568.40 | 1399.42 | 1398.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 1606.00 | 1431.71 | 1415.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-01-31 15:15:00 | 1254.50 | 2024-02-19 09:15:00 | 1261.20 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-02-01 10:00:00 | 1250.72 | 2024-02-19 09:15:00 | 1261.20 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-02-01 15:00:00 | 1251.05 | 2024-02-19 09:15:00 | 1261.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-02-02 09:15:00 | 1241.45 | 2024-02-19 09:15:00 | 1261.20 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-02-02 11:00:00 | 1233.58 | 2024-02-19 09:15:00 | 1261.20 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-02-07 10:15:00 | 1239.00 | 2024-02-27 12:15:00 | 1191.77 | PARTIAL | 0.50 | 3.81% |
| SELL | retest2 | 2024-02-07 11:30:00 | 1237.00 | 2024-02-27 12:15:00 | 1188.18 | PARTIAL | 0.50 | 3.95% |
| SELL | retest2 | 2024-02-08 10:00:00 | 1237.28 | 2024-02-27 12:15:00 | 1188.50 | PARTIAL | 0.50 | 3.94% |
| SELL | retest2 | 2024-02-16 13:45:00 | 1243.08 | 2024-02-28 10:15:00 | 1179.38 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2024-02-20 10:30:00 | 1237.60 | 2024-02-28 10:15:00 | 1175.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 10:15:00 | 1239.00 | 2024-03-05 09:15:00 | 1129.05 | TARGET_HIT | 0.50 | 8.87% |
| SELL | retest2 | 2024-02-07 11:30:00 | 1237.00 | 2024-03-05 10:15:00 | 1125.65 | TARGET_HIT | 0.50 | 9.00% |
| SELL | retest2 | 2024-02-08 10:00:00 | 1237.28 | 2024-03-05 10:15:00 | 1125.94 | TARGET_HIT | 0.50 | 9.00% |
| SELL | retest2 | 2024-02-16 13:45:00 | 1243.08 | 2024-03-05 10:15:00 | 1117.31 | TARGET_HIT | 0.50 | 10.12% |
| SELL | retest2 | 2024-02-20 10:30:00 | 1237.60 | 2024-03-05 11:15:00 | 1113.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-13 10:15:00 | 1240.00 | 2024-05-15 09:15:00 | 1249.68 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-13 11:30:00 | 1241.78 | 2024-05-15 09:15:00 | 1249.68 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-10-17 15:15:00 | 1665.90 | 2024-10-23 14:15:00 | 1629.58 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-10-22 11:30:00 | 1665.63 | 2024-10-23 14:15:00 | 1629.58 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-10-22 12:00:00 | 1667.30 | 2024-10-23 14:15:00 | 1629.58 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-10-22 13:45:00 | 1666.15 | 2024-10-23 14:15:00 | 1629.58 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-01-07 14:15:00 | 1545.00 | 2025-01-10 10:15:00 | 1467.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1537.53 | 2025-01-10 11:15:00 | 1460.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 14:15:00 | 1545.00 | 2025-01-15 09:15:00 | 1390.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1537.53 | 2025-01-15 09:15:00 | 1383.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-24 09:45:00 | 1408.75 | 2025-06-24 11:15:00 | 1391.35 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-24 10:15:00 | 1409.40 | 2025-06-24 11:15:00 | 1391.35 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-04 10:45:00 | 1414.55 | 2025-07-24 13:15:00 | 1553.09 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2025-07-04 14:00:00 | 1411.90 | 2025-07-25 09:15:00 | 1556.01 | TARGET_HIT | 1.00 | 10.21% |
| BUY | retest2 | 2025-10-13 14:15:00 | 1608.70 | 2025-10-15 09:15:00 | 1565.50 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-11-04 11:00:00 | 1573.90 | 2025-11-13 12:15:00 | 1585.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-04 12:30:00 | 1575.25 | 2025-11-13 12:15:00 | 1585.40 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-04 14:45:00 | 1575.00 | 2025-11-13 12:15:00 | 1585.40 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1564.55 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2025-11-06 10:30:00 | 1551.55 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2025-11-06 15:00:00 | 1546.30 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2025-11-10 12:30:00 | 1556.55 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2025-11-21 09:30:00 | 1558.20 | 2025-12-03 11:15:00 | 1484.09 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-03 11:15:00 | 1486.56 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-03 11:15:00 | 1488.27 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-03 11:15:00 | 1487.61 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-08 15:15:00 | 1480.29 | PARTIAL | 0.50 | 5.47% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-08 15:15:00 | 1477.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-08 15:15:00 | 1477.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-08 15:15:00 | 1477.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-08 15:15:00 | 1477.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:30:00 | 1558.20 | 2025-12-17 14:15:00 | 1402.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-17 14:15:00 | 1405.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-17 14:15:00 | 1408.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-17 14:15:00 | 1409.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-17 14:15:00 | 1409.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-17 14:15:00 | 1399.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-17 14:15:00 | 1399.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-17 14:15:00 | 1400.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-17 14:15:00 | 1399.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1488.60 | 2026-01-01 13:15:00 | 1507.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1486.60 | 2026-01-05 15:15:00 | 1507.40 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-02 09:45:00 | 1487.50 | 2026-01-05 15:15:00 | 1507.40 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-05 10:45:00 | 1490.00 | 2026-01-05 15:15:00 | 1507.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-06 10:30:00 | 1492.00 | 2026-01-09 09:15:00 | 1417.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:30:00 | 1492.00 | 2026-02-01 11:15:00 | 1425.90 | STOP_HIT | 0.50 | 4.43% |
