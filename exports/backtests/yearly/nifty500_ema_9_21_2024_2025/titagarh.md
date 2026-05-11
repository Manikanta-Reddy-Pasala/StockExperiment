# Titagarh Rail Systems Ltd. (TITAGARH)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 840.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 88 |
| ALERT2 | 87 |
| ALERT2_SKIP | 50 |
| ALERT3 | 212 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 108 |
| PARTIAL | 27 |
| TARGET_HIT | 16 |
| STOP_HIT | 96 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 76 / 63
- **Target hits / Stop hits / Partials:** 16 / 96 / 27
- **Avg / median % per leg:** 1.27% / 0.34%
- **Sum % (uncompounded):** 176.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 12 | 30.8% | 1 | 38 | 0 | -1.09% | -42.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 39 | 12 | 30.8% | 1 | 38 | 0 | -1.09% | -42.5% |
| SELL (all) | 100 | 64 | 64.0% | 15 | 58 | 27 | 2.19% | 218.9% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.68% | -18.7% |
| SELL @ 3rd Alert (retest2) | 96 | 64 | 66.7% | 15 | 54 | 27 | 2.47% | 237.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.68% | -18.7% |
| retest2 (combined) | 135 | 76 | 56.3% | 16 | 92 | 27 | 1.45% | 195.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1113.00 | 1090.01 | 1087.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 13:15:00 | 1131.00 | 1110.01 | 1098.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 1210.15 | 1226.38 | 1191.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 1210.15 | 1226.38 | 1191.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1225.05 | 1241.88 | 1221.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 1225.05 | 1241.88 | 1221.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 1242.25 | 1241.96 | 1223.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:30:00 | 1224.25 | 1241.96 | 1223.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1253.85 | 1245.45 | 1229.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 1227.85 | 1245.45 | 1229.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 1239.70 | 1242.79 | 1233.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:45:00 | 1233.90 | 1242.79 | 1233.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1287.10 | 1251.71 | 1239.83 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 13:15:00 | 1235.90 | 1243.52 | 1243.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1229.50 | 1240.71 | 1242.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 1252.00 | 1241.22 | 1242.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 1252.00 | 1241.22 | 1242.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1252.00 | 1241.22 | 1242.17 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 1257.20 | 1245.54 | 1244.00 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 1233.60 | 1242.38 | 1243.13 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 1253.75 | 1244.73 | 1243.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 15:15:00 | 1256.00 | 1246.99 | 1245.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1426.00 | 1461.46 | 1424.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1426.00 | 1461.46 | 1424.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1426.00 | 1461.46 | 1424.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1423.15 | 1461.46 | 1424.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1247.65 | 1418.70 | 1408.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1247.65 | 1418.70 | 1408.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1194.00 | 1373.76 | 1389.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 1044.90 | 1232.82 | 1308.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1204.10 | 1152.25 | 1217.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1204.10 | 1152.25 | 1217.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1204.10 | 1152.25 | 1217.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:15:00 | 1157.35 | 1163.13 | 1211.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:45:00 | 1160.00 | 1162.40 | 1206.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:15:00 | 1159.45 | 1163.22 | 1203.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 09:15:00 | 1311.95 | 1217.18 | 1208.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 09:15:00 | 1311.95 | 1217.18 | 1208.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 13:15:00 | 1332.05 | 1275.85 | 1242.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 14:15:00 | 1490.40 | 1491.74 | 1467.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 14:45:00 | 1492.30 | 1491.74 | 1467.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1486.00 | 1492.10 | 1471.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1466.85 | 1492.10 | 1471.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 1483.00 | 1492.21 | 1479.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 1483.00 | 1492.21 | 1479.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1482.90 | 1489.21 | 1480.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 15:15:00 | 1496.00 | 1484.17 | 1480.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-24 09:15:00 | 1645.60 | 1568.36 | 1532.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 09:15:00 | 1771.00 | 1806.02 | 1806.65 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 1823.00 | 1790.78 | 1790.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 1849.50 | 1819.86 | 1806.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 1801.00 | 1816.09 | 1805.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 10:15:00 | 1801.00 | 1816.09 | 1805.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1801.00 | 1816.09 | 1805.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1801.00 | 1816.09 | 1805.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1804.35 | 1813.74 | 1805.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 14:30:00 | 1829.10 | 1817.52 | 1809.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 15:15:00 | 1829.00 | 1817.52 | 1809.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 1786.95 | 1811.99 | 1809.25 | SL hit (close<static) qty=1.00 sl=1801.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 1781.20 | 1805.84 | 1806.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 1723.55 | 1782.13 | 1794.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 14:15:00 | 1699.95 | 1680.08 | 1697.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 14:15:00 | 1699.95 | 1680.08 | 1697.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1699.95 | 1680.08 | 1697.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:30:00 | 1733.60 | 1680.08 | 1697.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1710.00 | 1686.06 | 1698.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 1687.35 | 1686.06 | 1698.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1703.55 | 1689.56 | 1698.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:15:00 | 1680.75 | 1691.24 | 1697.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 1656.25 | 1691.52 | 1695.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 1596.71 | 1626.20 | 1654.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 1665.00 | 1623.53 | 1642.82 | SL hit (close>ema200) qty=0.50 sl=1623.53 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1635.00 | 1625.15 | 1624.97 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 1622.15 | 1624.55 | 1624.72 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 1639.00 | 1627.20 | 1625.88 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 1608.60 | 1623.48 | 1624.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 13:15:00 | 1596.45 | 1612.38 | 1618.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 10:15:00 | 1611.20 | 1602.40 | 1610.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 10:15:00 | 1611.20 | 1602.40 | 1610.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1611.20 | 1602.40 | 1610.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 1615.00 | 1602.40 | 1610.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 1615.50 | 1605.02 | 1611.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 1615.50 | 1605.02 | 1611.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 1603.20 | 1604.65 | 1610.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 14:00:00 | 1598.10 | 1603.34 | 1609.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 1667.00 | 1618.26 | 1614.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 1667.00 | 1618.26 | 1614.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 1691.65 | 1632.94 | 1621.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 1669.60 | 1670.16 | 1652.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 11:45:00 | 1669.25 | 1670.16 | 1652.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1611.90 | 1661.83 | 1655.83 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 10:15:00 | 1608.00 | 1651.06 | 1651.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 1598.60 | 1618.84 | 1631.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1447.20 | 1435.58 | 1483.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1447.20 | 1435.58 | 1483.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1447.20 | 1435.58 | 1483.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:30:00 | 1413.05 | 1429.90 | 1468.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 10:15:00 | 1342.40 | 1359.07 | 1383.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-09 15:15:00 | 1357.00 | 1353.96 | 1371.02 | SL hit (close>ema200) qty=0.50 sl=1353.96 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 1431.25 | 1378.63 | 1377.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 1453.65 | 1393.64 | 1384.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 1413.95 | 1416.97 | 1403.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 12:15:00 | 1413.95 | 1416.97 | 1403.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1413.95 | 1416.97 | 1403.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 1409.50 | 1416.97 | 1403.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1406.80 | 1414.93 | 1403.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1406.80 | 1414.93 | 1403.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1400.00 | 1411.95 | 1403.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:30:00 | 1400.20 | 1411.95 | 1403.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1399.05 | 1409.37 | 1402.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1398.35 | 1409.37 | 1402.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1403.95 | 1408.35 | 1403.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:00:00 | 1415.65 | 1409.81 | 1404.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:45:00 | 1424.45 | 1411.85 | 1406.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 1416.75 | 1412.51 | 1408.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 14:45:00 | 1419.40 | 1413.42 | 1410.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1446.50 | 1455.04 | 1441.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-21 14:15:00 | 1436.95 | 1441.18 | 1441.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 1436.95 | 1441.18 | 1441.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 1429.20 | 1437.63 | 1439.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 1429.45 | 1413.60 | 1420.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 12:15:00 | 1429.45 | 1413.60 | 1420.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1429.45 | 1413.60 | 1420.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 1429.45 | 1413.60 | 1420.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 1429.30 | 1416.74 | 1421.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 14:15:00 | 1425.90 | 1416.74 | 1421.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 1468.75 | 1431.35 | 1427.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 1468.75 | 1431.35 | 1427.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 10:15:00 | 1495.70 | 1444.22 | 1433.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 1475.50 | 1477.37 | 1458.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 09:30:00 | 1468.95 | 1477.37 | 1458.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1463.00 | 1470.54 | 1462.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 1455.95 | 1470.54 | 1462.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1451.25 | 1466.68 | 1461.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 1451.25 | 1466.68 | 1461.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1440.20 | 1461.39 | 1459.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 1440.15 | 1461.39 | 1459.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 1431.95 | 1455.50 | 1457.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1425.00 | 1446.38 | 1452.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 1425.95 | 1411.63 | 1421.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 1425.95 | 1411.63 | 1421.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1425.95 | 1411.63 | 1421.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 1425.95 | 1411.63 | 1421.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 1426.95 | 1414.69 | 1422.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:45:00 | 1432.00 | 1414.69 | 1422.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 1428.75 | 1417.50 | 1422.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:30:00 | 1426.00 | 1417.50 | 1422.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 1447.35 | 1423.47 | 1424.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:45:00 | 1451.30 | 1423.47 | 1424.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 1447.05 | 1428.19 | 1426.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 14:15:00 | 1451.00 | 1432.75 | 1429.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 1442.10 | 1444.23 | 1436.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 11:45:00 | 1440.20 | 1444.23 | 1436.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 1426.55 | 1443.35 | 1438.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 1426.55 | 1443.35 | 1438.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 1440.00 | 1442.68 | 1438.54 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 11:15:00 | 1426.65 | 1434.47 | 1435.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 1412.40 | 1426.50 | 1431.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1387.55 | 1385.44 | 1397.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1387.55 | 1385.44 | 1397.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1387.55 | 1385.44 | 1397.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 1373.90 | 1387.24 | 1393.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:30:00 | 1367.10 | 1352.80 | 1363.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 13:15:00 | 1305.20 | 1322.89 | 1336.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:15:00 | 1298.74 | 1309.05 | 1326.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-19 10:15:00 | 1236.51 | 1271.79 | 1295.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 1315.50 | 1291.04 | 1289.00 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 1294.85 | 1302.26 | 1302.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 15:15:00 | 1293.80 | 1299.40 | 1300.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 1171.50 | 1157.87 | 1181.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 1171.50 | 1157.87 | 1181.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1171.50 | 1157.87 | 1181.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 1171.50 | 1157.87 | 1181.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 1085.10 | 1089.05 | 1112.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 1108.65 | 1089.05 | 1112.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 1110.30 | 1093.91 | 1110.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 1110.30 | 1093.91 | 1110.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1111.75 | 1097.48 | 1110.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:30:00 | 1105.00 | 1097.48 | 1110.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1113.50 | 1100.68 | 1111.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1118.60 | 1100.68 | 1111.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1119.00 | 1104.35 | 1111.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 1108.85 | 1106.46 | 1111.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 1132.05 | 1102.36 | 1103.96 | SL hit (close>static) qty=1.00 sl=1128.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 1146.70 | 1105.26 | 1103.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 10:15:00 | 1156.05 | 1115.42 | 1108.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 12:15:00 | 1111.35 | 1115.62 | 1109.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 12:15:00 | 1111.35 | 1115.62 | 1109.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 1111.35 | 1115.62 | 1109.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:00:00 | 1111.35 | 1115.62 | 1109.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1116.85 | 1115.87 | 1110.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:15:00 | 1125.30 | 1116.70 | 1111.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:45:00 | 1124.40 | 1118.77 | 1113.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:30:00 | 1131.00 | 1121.51 | 1115.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 1121.25 | 1130.44 | 1123.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1123.50 | 1129.05 | 1123.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 1128.60 | 1126.26 | 1123.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 14:15:00 | 1138.25 | 1160.26 | 1162.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 1138.25 | 1160.26 | 1162.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 1132.05 | 1153.51 | 1158.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1158.55 | 1129.97 | 1142.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 1158.55 | 1129.97 | 1142.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1158.55 | 1129.97 | 1142.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 1158.55 | 1129.97 | 1142.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1178.35 | 1139.64 | 1145.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 1178.35 | 1139.64 | 1145.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 1174.00 | 1152.14 | 1150.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 13:15:00 | 1180.05 | 1157.72 | 1153.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 12:15:00 | 1172.00 | 1172.08 | 1164.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-24 12:45:00 | 1168.00 | 1172.08 | 1164.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1163.95 | 1169.83 | 1164.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:45:00 | 1164.60 | 1169.83 | 1164.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1172.00 | 1170.26 | 1165.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 1154.35 | 1170.26 | 1165.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1123.10 | 1160.83 | 1161.46 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1176.95 | 1154.73 | 1152.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 1207.80 | 1169.25 | 1160.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1192.50 | 1212.75 | 1197.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1192.50 | 1212.75 | 1197.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1192.50 | 1212.75 | 1197.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1192.50 | 1212.75 | 1197.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1193.15 | 1208.83 | 1197.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 1202.70 | 1208.83 | 1197.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 1203.90 | 1207.30 | 1198.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 14:15:00 | 1176.00 | 1198.25 | 1195.90 | SL hit (close<static) qty=1.00 sl=1188.20 alert=retest2 |

### Cycle 30 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 1175.55 | 1193.71 | 1194.05 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1208.40 | 1192.69 | 1192.03 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 1183.95 | 1202.41 | 1202.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1172.65 | 1192.41 | 1197.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 1185.50 | 1166.14 | 1175.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 14:15:00 | 1185.50 | 1166.14 | 1175.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 1185.50 | 1166.14 | 1175.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:45:00 | 1189.80 | 1166.14 | 1175.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 1182.00 | 1169.31 | 1176.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 1180.20 | 1169.31 | 1176.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1169.00 | 1169.30 | 1174.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 1165.10 | 1167.98 | 1173.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 11:15:00 | 1106.84 | 1133.81 | 1152.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 1111.10 | 1110.22 | 1131.80 | SL hit (close>ema200) qty=0.50 sl=1110.22 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1127.35 | 1116.22 | 1116.18 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 1111.30 | 1115.74 | 1116.00 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 15:15:00 | 1119.95 | 1116.58 | 1116.36 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 1081.55 | 1109.57 | 1113.19 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1170.75 | 1111.79 | 1106.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 1185.65 | 1162.79 | 1150.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 1211.40 | 1216.97 | 1199.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:15:00 | 1203.50 | 1216.97 | 1199.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1200.95 | 1213.77 | 1199.52 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 1189.00 | 1198.12 | 1198.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 09:15:00 | 1185.95 | 1194.24 | 1196.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 15:15:00 | 1188.95 | 1188.48 | 1192.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:15:00 | 1218.95 | 1188.48 | 1192.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1213.85 | 1193.55 | 1194.12 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 1213.55 | 1197.55 | 1195.89 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 1183.00 | 1196.86 | 1197.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 09:15:00 | 1179.00 | 1188.47 | 1192.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 1193.00 | 1189.38 | 1192.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 10:15:00 | 1193.00 | 1189.38 | 1192.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1193.00 | 1189.38 | 1192.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 1193.00 | 1189.38 | 1192.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 1188.65 | 1189.23 | 1191.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 12:30:00 | 1186.50 | 1189.13 | 1191.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 13:15:00 | 1187.65 | 1189.13 | 1191.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 1200.20 | 1191.05 | 1191.59 | SL hit (close>static) qty=1.00 sl=1200.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 1209.55 | 1194.75 | 1193.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 11:15:00 | 1214.25 | 1198.65 | 1195.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 1298.35 | 1317.98 | 1296.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 10:00:00 | 1298.35 | 1317.98 | 1296.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1309.75 | 1316.33 | 1297.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:45:00 | 1313.85 | 1315.62 | 1298.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 1320.85 | 1311.25 | 1301.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 1286.05 | 1314.26 | 1317.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 1286.05 | 1314.26 | 1317.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1258.85 | 1287.67 | 1300.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 15:15:00 | 1280.00 | 1275.34 | 1287.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 09:15:00 | 1281.10 | 1275.34 | 1287.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1264.45 | 1273.16 | 1285.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:45:00 | 1259.80 | 1281.21 | 1284.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 1248.90 | 1269.32 | 1276.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:15:00 | 1196.81 | 1223.33 | 1244.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:15:00 | 1186.45 | 1223.33 | 1244.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-30 09:15:00 | 1133.82 | 1150.55 | 1178.65 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 11:15:00 | 1129.15 | 1120.49 | 1119.61 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1109.00 | 1118.02 | 1118.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1090.65 | 1112.54 | 1116.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1104.90 | 1098.42 | 1105.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1104.90 | 1098.42 | 1105.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1104.90 | 1098.42 | 1105.83 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 1127.00 | 1111.53 | 1110.59 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 1103.15 | 1111.89 | 1112.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 1098.95 | 1108.07 | 1110.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 14:15:00 | 1008.95 | 1007.19 | 1024.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:45:00 | 1012.80 | 1007.19 | 1024.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1022.00 | 1010.76 | 1023.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:15:00 | 1009.40 | 1018.10 | 1022.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 1065.85 | 1026.26 | 1025.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1065.85 | 1026.26 | 1025.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 1107.75 | 1057.11 | 1042.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1071.05 | 1082.06 | 1072.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1071.05 | 1082.06 | 1072.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1071.05 | 1082.06 | 1072.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1071.05 | 1082.06 | 1072.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1065.60 | 1078.77 | 1072.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 1070.80 | 1078.77 | 1072.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1075.00 | 1078.01 | 1072.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 1074.65 | 1078.01 | 1072.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1068.65 | 1076.14 | 1072.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 1068.65 | 1076.14 | 1072.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1066.95 | 1074.30 | 1071.61 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 1050.00 | 1066.83 | 1068.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1018.65 | 1057.19 | 1063.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 941.00 | 937.05 | 962.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 941.00 | 937.05 | 962.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 939.45 | 931.77 | 944.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 939.45 | 931.77 | 944.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 950.70 | 938.67 | 943.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 950.70 | 938.67 | 943.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 950.55 | 941.04 | 943.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 970.95 | 941.04 | 943.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 997.95 | 952.43 | 948.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 1013.70 | 981.18 | 967.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 954.00 | 1017.86 | 999.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 954.00 | 1017.86 | 999.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 954.00 | 1017.86 | 999.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 959.45 | 1017.86 | 999.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 960.00 | 1006.29 | 996.12 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 950.00 | 987.53 | 988.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 894.00 | 968.82 | 980.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 925.40 | 922.59 | 944.99 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:45:00 | 894.75 | 918.37 | 941.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:30:00 | 897.55 | 914.98 | 937.43 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 13:45:00 | 898.35 | 910.11 | 931.21 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 948.90 | 915.35 | 928.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 948.90 | 915.35 | 928.11 | SL hit (close>ema400) qty=1.00 sl=928.11 alert=retest1 |

### Cycle 51 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 806.35 | 791.34 | 789.80 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 784.75 | 793.78 | 794.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 780.95 | 789.65 | 792.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 775.50 | 775.39 | 781.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 775.50 | 775.39 | 781.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 775.50 | 775.39 | 781.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 760.80 | 775.27 | 778.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 722.76 | 741.30 | 756.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 09:15:00 | 684.72 | 703.54 | 726.79 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 716.75 | 703.60 | 702.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 724.50 | 709.77 | 705.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 729.25 | 746.88 | 735.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 729.25 | 746.88 | 735.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 729.25 | 746.88 | 735.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 729.25 | 746.88 | 735.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 726.00 | 742.70 | 734.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 726.00 | 742.70 | 734.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 711.80 | 728.34 | 729.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 704.00 | 723.47 | 727.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 700.35 | 697.39 | 705.50 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 09:15:00 | 689.95 | 698.23 | 704.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 704.10 | 699.40 | 704.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 704.10 | 699.40 | 704.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 701.10 | 699.74 | 704.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 696.20 | 699.39 | 703.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:15:00 | 696.60 | 699.39 | 703.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:30:00 | 696.15 | 698.54 | 702.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 696.85 | 698.12 | 701.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 699.00 | 694.56 | 697.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 699.00 | 694.56 | 697.62 | SL hit (close>ema400) qty=1.00 sl=697.62 alert=retest1 |

### Cycle 55 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 710.25 | 699.82 | 699.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 721.50 | 706.10 | 702.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 803.30 | 809.66 | 791.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:00:00 | 803.30 | 809.66 | 791.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 809.65 | 812.60 | 801.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 809.05 | 812.60 | 801.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 804.85 | 809.97 | 802.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 804.85 | 809.97 | 802.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 802.65 | 808.51 | 802.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 802.65 | 808.51 | 802.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 805.00 | 807.80 | 802.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 803.45 | 807.80 | 802.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 805.00 | 807.24 | 802.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 805.00 | 807.24 | 802.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 801.50 | 806.09 | 802.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 798.10 | 806.09 | 802.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 795.55 | 803.99 | 802.00 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 790.50 | 800.42 | 800.67 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 803.90 | 801.12 | 800.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 813.00 | 804.77 | 802.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 803.05 | 810.07 | 806.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 12:15:00 | 803.05 | 810.07 | 806.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 803.05 | 810.07 | 806.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 803.05 | 810.07 | 806.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 800.00 | 808.06 | 806.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 800.00 | 808.06 | 806.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 798.85 | 806.22 | 805.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 798.85 | 806.22 | 805.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 793.00 | 803.57 | 804.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 787.40 | 798.03 | 800.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 800.80 | 797.96 | 800.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 800.80 | 797.96 | 800.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 800.80 | 797.96 | 800.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 800.80 | 797.96 | 800.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 808.05 | 799.97 | 800.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 808.05 | 799.97 | 800.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 807.45 | 801.47 | 801.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 807.25 | 801.47 | 801.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 808.95 | 802.97 | 802.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 817.90 | 808.19 | 804.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 785.50 | 811.92 | 809.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 785.50 | 811.92 | 809.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 785.50 | 811.92 | 809.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 785.50 | 811.92 | 809.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 798.70 | 809.27 | 808.42 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 793.50 | 806.12 | 807.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 789.40 | 800.37 | 804.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 762.45 | 753.83 | 771.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 778.60 | 753.83 | 771.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 762.15 | 755.49 | 770.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 754.35 | 756.85 | 769.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 745.60 | 759.37 | 766.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:30:00 | 755.00 | 745.57 | 753.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 12:15:00 | 754.00 | 749.43 | 753.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 753.00 | 750.14 | 753.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:15:00 | 748.45 | 750.14 | 753.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 783.00 | 756.14 | 755.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 783.00 | 756.14 | 755.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 788.15 | 762.54 | 758.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 785.20 | 786.88 | 779.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 785.20 | 786.88 | 779.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 790.35 | 795.88 | 790.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 15:00:00 | 790.35 | 795.88 | 790.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 791.35 | 794.98 | 790.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:15:00 | 803.50 | 794.98 | 790.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 801.30 | 796.24 | 791.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 814.95 | 797.63 | 795.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:15:00 | 812.80 | 811.45 | 804.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 774.60 | 800.81 | 801.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 774.60 | 800.81 | 801.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 13:15:00 | 768.00 | 774.76 | 780.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 703.00 | 700.24 | 711.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 707.75 | 700.24 | 711.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 723.50 | 704.89 | 713.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 723.50 | 704.89 | 713.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 719.35 | 707.78 | 713.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:45:00 | 714.60 | 710.32 | 713.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 678.87 | 697.83 | 706.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 691.00 | 687.50 | 696.18 | SL hit (close>ema200) qty=0.50 sl=687.50 alert=retest2 |

### Cycle 63 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 740.00 | 705.32 | 703.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 741.35 | 717.71 | 709.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 15:15:00 | 935.65 | 935.89 | 896.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 894.00 | 935.89 | 896.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 895.55 | 927.82 | 896.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 895.80 | 927.82 | 896.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 904.00 | 923.06 | 897.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 920.40 | 910.03 | 899.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 916.30 | 913.61 | 904.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 892.45 | 903.35 | 903.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 892.45 | 903.35 | 903.80 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 906.05 | 902.84 | 902.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 919.25 | 909.63 | 906.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 10:15:00 | 923.30 | 925.66 | 918.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 923.30 | 925.66 | 918.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 923.30 | 925.66 | 918.48 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 908.25 | 916.34 | 916.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 895.55 | 909.83 | 912.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 909.40 | 899.58 | 904.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 909.40 | 899.58 | 904.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 909.40 | 899.58 | 904.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 909.40 | 899.58 | 904.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 904.10 | 900.48 | 904.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 898.00 | 900.48 | 904.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 901.10 | 901.07 | 904.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 13:15:00 | 912.90 | 903.39 | 904.96 | SL hit (close>static) qty=1.00 sl=909.80 alert=retest2 |

### Cycle 67 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 929.60 | 904.48 | 903.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 934.85 | 910.55 | 906.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 930.70 | 932.14 | 925.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 930.70 | 932.14 | 925.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 941.70 | 949.67 | 944.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 941.45 | 949.67 | 944.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 942.45 | 948.22 | 944.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 942.45 | 948.22 | 944.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 942.00 | 946.75 | 944.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 936.95 | 946.75 | 944.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 946.15 | 946.63 | 944.64 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 929.45 | 942.20 | 943.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 906.25 | 928.09 | 935.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 897.90 | 897.22 | 906.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 897.90 | 897.22 | 906.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 865.50 | 852.30 | 863.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 865.50 | 852.30 | 863.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 863.85 | 854.61 | 863.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 859.00 | 854.61 | 863.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 860.00 | 857.35 | 863.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 883.30 | 865.48 | 865.87 | SL hit (close>static) qty=1.00 sl=874.40 alert=retest2 |

### Cycle 69 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 885.10 | 869.41 | 867.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 888.50 | 873.23 | 869.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 936.05 | 943.04 | 933.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 936.05 | 943.04 | 933.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 937.00 | 941.84 | 933.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 951.00 | 941.84 | 933.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 943.60 | 947.96 | 948.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 943.60 | 947.96 | 948.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 941.25 | 946.62 | 947.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 944.00 | 938.76 | 942.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 944.00 | 938.76 | 942.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 944.00 | 938.76 | 942.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 944.00 | 938.76 | 942.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 941.50 | 939.30 | 942.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 942.95 | 939.30 | 942.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 939.05 | 939.25 | 941.80 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 949.00 | 943.41 | 943.23 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 938.10 | 943.37 | 943.62 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 946.50 | 943.46 | 943.42 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 936.00 | 943.76 | 944.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 929.25 | 935.79 | 939.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 934.55 | 934.50 | 937.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 13:45:00 | 934.70 | 934.50 | 937.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 938.65 | 935.33 | 937.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 938.65 | 935.33 | 937.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 937.90 | 935.85 | 937.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 944.50 | 935.85 | 937.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 940.50 | 936.78 | 937.97 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 942.85 | 938.84 | 938.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 946.40 | 940.35 | 939.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 946.05 | 949.55 | 946.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 946.05 | 949.55 | 946.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 946.05 | 949.55 | 946.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 946.05 | 949.55 | 946.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 944.15 | 948.47 | 946.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 15:00:00 | 947.10 | 947.72 | 946.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 936.60 | 945.06 | 945.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 936.60 | 945.06 | 945.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 930.70 | 939.47 | 942.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 938.20 | 933.07 | 936.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 938.20 | 933.07 | 936.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 938.20 | 933.07 | 936.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 940.00 | 933.07 | 936.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 935.60 | 933.58 | 936.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 934.25 | 935.76 | 936.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 887.54 | 900.19 | 910.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 896.75 | 890.47 | 901.20 | SL hit (close>ema200) qty=0.50 sl=890.47 alert=retest2 |

### Cycle 77 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 862.20 | 856.42 | 856.16 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 853.10 | 855.76 | 855.88 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 858.15 | 856.23 | 856.09 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 841.35 | 853.41 | 854.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 836.00 | 842.88 | 847.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 843.20 | 842.94 | 847.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 12:00:00 | 843.20 | 842.94 | 847.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 854.00 | 842.86 | 845.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 854.00 | 842.86 | 845.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 856.15 | 845.51 | 846.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 845.05 | 845.51 | 846.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 14:15:00 | 802.80 | 825.90 | 835.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 802.10 | 790.92 | 807.05 | SL hit (close>ema200) qty=0.50 sl=790.92 alert=retest2 |

### Cycle 81 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 813.05 | 807.11 | 806.98 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 804.00 | 806.95 | 807.33 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 808.60 | 807.52 | 807.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 820.45 | 810.66 | 808.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 818.50 | 819.06 | 815.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 819.10 | 819.06 | 815.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 859.55 | 860.97 | 855.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 849.50 | 860.97 | 855.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 852.60 | 863.19 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 852.00 | 863.19 | 859.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 851.75 | 860.90 | 859.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 851.75 | 860.90 | 859.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 847.85 | 857.35 | 857.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 845.20 | 854.92 | 856.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 837.00 | 835.31 | 842.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 837.00 | 835.31 | 842.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 837.00 | 835.31 | 842.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 839.10 | 835.31 | 842.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 842.60 | 837.87 | 841.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 843.00 | 837.87 | 841.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 842.70 | 838.83 | 841.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 841.00 | 838.83 | 841.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 850.00 | 842.72 | 842.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 851.95 | 842.72 | 842.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 853.00 | 844.78 | 843.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 859.85 | 847.79 | 845.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 850.05 | 850.23 | 847.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 850.05 | 850.23 | 847.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 850.05 | 850.23 | 847.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 849.50 | 850.23 | 847.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 848.60 | 849.91 | 847.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 852.95 | 848.43 | 847.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 844.55 | 847.65 | 847.37 | SL hit (close<static) qty=1.00 sl=847.10 alert=retest2 |

### Cycle 86 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 844.75 | 847.07 | 847.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 841.70 | 846.00 | 846.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 842.50 | 840.91 | 842.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 842.50 | 840.91 | 842.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 842.50 | 840.91 | 842.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 843.25 | 840.91 | 842.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 839.95 | 840.72 | 842.64 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 849.00 | 844.05 | 843.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 857.00 | 848.22 | 846.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 903.85 | 904.11 | 891.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 903.85 | 904.11 | 891.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 931.20 | 939.04 | 929.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 930.75 | 939.04 | 929.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 934.65 | 936.90 | 930.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 940.85 | 936.90 | 930.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:00:00 | 937.70 | 937.06 | 930.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 937.15 | 944.46 | 944.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 937.15 | 944.46 | 944.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 922.35 | 935.76 | 940.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 911.75 | 909.27 | 919.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 911.75 | 909.27 | 919.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 911.75 | 909.27 | 919.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 889.35 | 903.88 | 911.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 890.25 | 900.94 | 909.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 892.10 | 889.46 | 898.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:30:00 | 893.40 | 890.80 | 898.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 882.00 | 877.91 | 881.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 882.00 | 877.91 | 881.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 886.00 | 879.53 | 881.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 888.25 | 879.53 | 881.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 883.00 | 881.12 | 882.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 880.60 | 881.12 | 882.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 886.00 | 882.09 | 882.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 886.00 | 882.09 | 882.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 889.35 | 883.54 | 883.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 889.35 | 883.54 | 883.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 890.95 | 886.17 | 884.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 882.95 | 885.53 | 884.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 882.95 | 885.53 | 884.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 882.95 | 885.53 | 884.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 885.95 | 885.53 | 884.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 879.45 | 884.31 | 883.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 879.30 | 884.31 | 883.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 885.80 | 884.61 | 884.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 887.40 | 884.89 | 884.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:15:00 | 886.90 | 885.17 | 884.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 889.90 | 900.54 | 901.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 889.90 | 900.54 | 901.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 882.50 | 888.03 | 892.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 891.30 | 887.80 | 891.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 891.30 | 887.80 | 891.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 891.30 | 887.80 | 891.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 891.30 | 887.80 | 891.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 890.70 | 888.38 | 891.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:45:00 | 888.05 | 889.85 | 891.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 888.40 | 889.50 | 890.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:45:00 | 886.20 | 880.85 | 880.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 887.00 | 882.08 | 881.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 887.00 | 882.08 | 881.46 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 878.00 | 881.14 | 881.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 876.30 | 880.34 | 880.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 879.25 | 878.47 | 879.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 879.25 | 878.47 | 879.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 879.25 | 878.47 | 879.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 879.25 | 878.47 | 879.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 878.10 | 878.39 | 879.67 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 890.15 | 880.75 | 880.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 894.70 | 886.36 | 883.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 892.60 | 895.07 | 890.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:45:00 | 893.85 | 895.07 | 890.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 896.45 | 895.35 | 890.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 889.95 | 895.35 | 890.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 891.15 | 894.32 | 891.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 900.00 | 894.32 | 891.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 894.75 | 897.88 | 898.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 894.75 | 897.88 | 898.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 883.50 | 895.01 | 896.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 912.60 | 896.92 | 897.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 912.60 | 896.92 | 897.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 912.60 | 896.92 | 897.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 912.60 | 896.92 | 897.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 908.00 | 899.14 | 898.34 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 889.05 | 898.04 | 899.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 883.50 | 895.13 | 897.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 849.80 | 848.04 | 858.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 849.80 | 848.04 | 858.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 855.40 | 849.61 | 854.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 855.40 | 849.61 | 854.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 849.75 | 849.63 | 854.00 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 867.60 | 856.52 | 856.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 886.95 | 863.31 | 859.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 872.60 | 876.19 | 869.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 872.60 | 876.19 | 869.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 872.60 | 876.19 | 869.78 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 863.70 | 867.39 | 867.85 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 871.80 | 868.33 | 868.20 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 861.10 | 866.88 | 867.55 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 893.80 | 872.07 | 869.62 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 869.70 | 873.05 | 873.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 869.10 | 872.26 | 872.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 877.10 | 870.64 | 871.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 877.10 | 870.64 | 871.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 877.10 | 870.64 | 871.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:45:00 | 875.10 | 870.64 | 871.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 878.40 | 872.19 | 872.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:30:00 | 872.80 | 872.37 | 872.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 873.65 | 872.63 | 872.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 873.65 | 872.63 | 872.58 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 869.60 | 872.02 | 872.30 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 874.30 | 872.45 | 872.41 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 866.25 | 871.28 | 871.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 856.50 | 867.56 | 870.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 857.45 | 856.45 | 862.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 857.45 | 856.45 | 862.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 846.65 | 842.56 | 847.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 846.20 | 842.56 | 847.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 844.05 | 842.86 | 847.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 844.05 | 842.86 | 847.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 847.20 | 843.73 | 847.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 847.20 | 843.73 | 847.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 844.15 | 843.81 | 847.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 843.00 | 843.81 | 847.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 848.05 | 845.12 | 847.30 | SL hit (close>static) qty=1.00 sl=847.60 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 772.45 | 768.73 | 768.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 774.00 | 770.47 | 769.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 788.65 | 789.05 | 783.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:30:00 | 789.20 | 789.05 | 783.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 784.95 | 788.60 | 784.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 784.75 | 788.60 | 784.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 782.20 | 787.32 | 784.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:15:00 | 780.70 | 787.32 | 784.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 781.60 | 786.18 | 784.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 781.60 | 786.18 | 784.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 775.95 | 782.07 | 782.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 774.45 | 780.54 | 781.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 776.50 | 776.09 | 779.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 776.50 | 776.09 | 779.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 770.70 | 773.21 | 776.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 774.70 | 773.21 | 776.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 777.90 | 772.07 | 774.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 777.90 | 772.07 | 774.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 785.15 | 774.69 | 775.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 785.15 | 774.69 | 775.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 784.05 | 776.56 | 776.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 801.55 | 781.56 | 778.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 850.75 | 851.23 | 835.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 873.05 | 851.23 | 835.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 890.25 | 900.69 | 886.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 922.70 | 902.81 | 888.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 908.05 | 903.73 | 891.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 878.20 | 897.10 | 890.65 | SL hit (close<static) qty=1.00 sl=886.45 alert=retest2 |

### Cycle 110 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 882.50 | 890.30 | 890.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 881.60 | 888.56 | 889.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 887.40 | 885.75 | 887.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 887.40 | 885.75 | 887.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 887.40 | 885.75 | 887.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:15:00 | 889.60 | 885.75 | 887.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 889.60 | 886.52 | 887.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 886.20 | 886.70 | 887.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:45:00 | 885.90 | 886.64 | 887.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 891.80 | 888.22 | 888.29 | SL hit (close>static) qty=1.00 sl=891.60 alert=retest2 |

### Cycle 111 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 823.80 | 807.63 | 806.02 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 801.00 | 806.65 | 807.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 789.85 | 801.58 | 804.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 782.50 | 774.38 | 780.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 782.50 | 774.38 | 780.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 782.50 | 774.38 | 780.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 793.80 | 774.38 | 780.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 786.80 | 776.86 | 781.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 781.60 | 779.32 | 781.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 782.90 | 779.28 | 781.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 796.60 | 783.53 | 783.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 796.60 | 783.53 | 783.15 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 777.35 | 782.87 | 783.59 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 792.00 | 783.60 | 783.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 807.75 | 790.55 | 786.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 804.65 | 804.93 | 796.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:45:00 | 802.45 | 804.93 | 796.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 814.05 | 806.83 | 801.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 817.05 | 808.87 | 802.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:45:00 | 816.90 | 810.43 | 804.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 820.65 | 811.90 | 805.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 817.25 | 813.67 | 807.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 813.50 | 820.54 | 813.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 813.50 | 820.54 | 813.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 798.75 | 816.18 | 811.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 800.80 | 816.18 | 811.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 792.95 | 811.54 | 810.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 792.95 | 811.54 | 810.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 785.40 | 806.31 | 807.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 785.40 | 806.31 | 807.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 776.25 | 794.66 | 801.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 791.95 | 787.28 | 795.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 791.95 | 787.28 | 795.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 794.30 | 788.68 | 795.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 799.50 | 788.68 | 795.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 795.50 | 790.05 | 795.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 790.25 | 796.32 | 796.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 792.55 | 796.56 | 796.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 794.00 | 794.62 | 795.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 788.45 | 783.60 | 783.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 788.45 | 783.60 | 783.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 794.90 | 787.46 | 785.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 785.10 | 790.89 | 788.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 785.10 | 790.89 | 788.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 785.10 | 790.89 | 788.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 785.20 | 790.89 | 788.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 780.65 | 788.84 | 787.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 779.20 | 788.84 | 787.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 778.65 | 785.38 | 786.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 770.70 | 779.88 | 783.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 764.25 | 758.55 | 765.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 764.25 | 758.55 | 765.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 764.25 | 758.55 | 765.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:30:00 | 768.20 | 758.55 | 765.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 757.15 | 758.18 | 762.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 757.15 | 758.18 | 762.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 763.20 | 759.18 | 763.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 763.20 | 759.18 | 763.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 763.00 | 759.95 | 763.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 768.10 | 759.95 | 763.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 761.65 | 760.29 | 762.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 759.85 | 761.95 | 762.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 721.86 | 730.10 | 736.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 683.87 | 700.19 | 710.30 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 119 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 686.80 | 669.25 | 667.42 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 651.40 | 667.83 | 668.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 640.80 | 655.47 | 658.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 662.15 | 656.30 | 658.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 662.15 | 656.30 | 658.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 662.15 | 656.30 | 658.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 662.15 | 656.30 | 658.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 660.75 | 657.19 | 658.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 656.00 | 657.40 | 658.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 651.20 | 657.82 | 658.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 623.20 | 639.14 | 647.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 618.64 | 639.14 | 647.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 629.95 | 627.80 | 637.54 | SL hit (close>ema200) qty=0.50 sl=627.80 alert=retest2 |

### Cycle 121 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 647.70 | 635.30 | 635.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 650.00 | 643.84 | 639.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 635.05 | 642.08 | 639.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 635.05 | 642.08 | 639.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 635.05 | 642.08 | 639.50 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 630.85 | 637.67 | 637.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 622.70 | 634.68 | 636.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 636.10 | 631.55 | 634.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 636.10 | 631.55 | 634.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 636.10 | 631.55 | 634.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 637.15 | 631.55 | 634.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 634.65 | 632.17 | 634.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 635.55 | 632.17 | 634.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 632.00 | 632.13 | 634.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 629.40 | 632.13 | 634.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 629.20 | 631.45 | 633.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 628.65 | 630.56 | 632.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 597.93 | 620.94 | 627.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 597.74 | 620.94 | 627.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 597.22 | 612.62 | 622.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 606.00 | 604.43 | 613.88 | SL hit (close>ema200) qty=0.50 sl=604.43 alert=retest2 |

### Cycle 123 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 637.60 | 618.44 | 617.48 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 602.75 | 620.24 | 621.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 600.70 | 611.18 | 616.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 613.10 | 588.73 | 596.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 613.10 | 588.73 | 596.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 613.10 | 588.73 | 596.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 617.00 | 588.73 | 596.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 608.40 | 592.66 | 597.89 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 617.05 | 601.14 | 601.06 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 595.20 | 600.71 | 601.30 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 611.80 | 603.11 | 602.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 614.45 | 605.38 | 603.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 707.55 | 708.62 | 693.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 707.10 | 708.62 | 693.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 690.80 | 707.67 | 703.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 699.45 | 705.74 | 702.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 699.25 | 705.74 | 702.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 700.85 | 702.20 | 701.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 697.10 | 701.18 | 701.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 697.10 | 701.18 | 701.36 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 718.90 | 704.06 | 702.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 11:15:00 | 733.95 | 719.28 | 712.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 738.20 | 739.45 | 731.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 738.20 | 739.45 | 731.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 738.20 | 739.45 | 731.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 747.15 | 740.92 | 738.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:45:00 | 746.50 | 740.97 | 738.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 727.25 | 736.64 | 737.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 727.25 | 736.64 | 737.66 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 755.10 | 738.66 | 737.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 15:15:00 | 760.00 | 742.92 | 739.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 789.75 | 790.27 | 780.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 789.75 | 790.27 | 780.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 786.30 | 788.42 | 781.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 773.00 | 788.42 | 781.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 763.50 | 783.43 | 779.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 763.50 | 783.43 | 779.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 764.00 | 779.55 | 778.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 759.00 | 779.55 | 778.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 766.75 | 776.99 | 777.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 761.95 | 768.98 | 771.58 | Break + close below crossover candle low |

### Cycle 133 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 796.00 | 774.38 | 773.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 863.85 | 792.28 | 781.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 835.30 | 836.87 | 821.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 835.30 | 836.87 | 821.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-06 12:15:00 | 1157.35 | 2024-06-10 09:15:00 | 1311.95 | STOP_HIT | 1.00 | -13.36% |
| SELL | retest2 | 2024-06-06 12:45:00 | 1160.00 | 2024-06-10 09:15:00 | 1311.95 | STOP_HIT | 1.00 | -13.10% |
| SELL | retest2 | 2024-06-06 14:15:00 | 1159.45 | 2024-06-10 09:15:00 | 1311.95 | STOP_HIT | 1.00 | -13.15% |
| BUY | retest2 | 2024-06-20 15:15:00 | 1496.00 | 2024-06-24 09:15:00 | 1645.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 14:30:00 | 1829.10 | 2024-07-09 10:15:00 | 1786.95 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-07-08 15:15:00 | 1829.00 | 2024-07-09 10:15:00 | 1786.95 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-07-16 13:15:00 | 1680.75 | 2024-07-19 09:15:00 | 1596.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 13:15:00 | 1680.75 | 2024-07-19 13:15:00 | 1665.00 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2024-07-18 09:15:00 | 1656.25 | 2024-07-23 12:15:00 | 1490.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-19 14:00:00 | 1665.00 | 2024-07-23 12:15:00 | 1498.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-26 14:00:00 | 1598.10 | 2024-07-29 09:15:00 | 1667.00 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2024-08-06 12:30:00 | 1413.05 | 2024-08-09 10:15:00 | 1342.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 12:30:00 | 1413.05 | 2024-08-09 15:15:00 | 1357.00 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2024-08-14 12:00:00 | 1415.65 | 2024-08-21 14:15:00 | 1436.95 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2024-08-14 12:45:00 | 1424.45 | 2024-08-21 14:15:00 | 1436.95 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2024-08-16 09:30:00 | 1416.75 | 2024-08-21 14:15:00 | 1436.95 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2024-08-16 14:45:00 | 1419.40 | 2024-08-21 14:15:00 | 1436.95 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-08-26 14:15:00 | 1425.90 | 2024-08-27 09:15:00 | 1468.75 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-09-11 10:00:00 | 1373.90 | 2024-09-17 13:15:00 | 1305.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 09:30:00 | 1367.10 | 2024-09-18 09:15:00 | 1298.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 10:00:00 | 1373.90 | 2024-09-19 10:15:00 | 1236.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-13 09:30:00 | 1367.10 | 2024-09-19 11:15:00 | 1230.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-09 11:30:00 | 1108.85 | 2024-10-11 09:15:00 | 1132.05 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-10-11 10:15:00 | 1109.00 | 2024-10-14 09:15:00 | 1146.70 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-10-14 15:15:00 | 1125.30 | 2024-10-21 14:15:00 | 1138.25 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2024-10-15 10:45:00 | 1124.40 | 2024-10-21 14:15:00 | 1138.25 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-10-15 11:30:00 | 1131.00 | 2024-10-21 14:15:00 | 1138.25 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2024-10-16 11:15:00 | 1121.25 | 2024-10-21 14:15:00 | 1138.25 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-10-16 14:15:00 | 1128.60 | 2024-10-21 14:15:00 | 1138.25 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-11-04 11:15:00 | 1202.70 | 2024-11-04 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-11-04 13:00:00 | 1203.90 | 2024-11-04 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-11-12 11:45:00 | 1165.10 | 2024-11-13 11:15:00 | 1106.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:45:00 | 1165.10 | 2024-11-14 09:15:00 | 1111.10 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2024-12-06 12:30:00 | 1186.50 | 2024-12-09 09:15:00 | 1200.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-12-06 13:15:00 | 1187.65 | 2024-12-09 09:15:00 | 1200.20 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-12-13 11:45:00 | 1313.85 | 2024-12-18 09:15:00 | 1286.05 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-12-16 09:15:00 | 1320.85 | 2024-12-18 09:15:00 | 1286.05 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-12-23 09:45:00 | 1259.80 | 2024-12-26 09:15:00 | 1196.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-23 15:15:00 | 1248.90 | 2024-12-26 09:15:00 | 1186.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-23 09:45:00 | 1259.80 | 2024-12-30 09:15:00 | 1133.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-23 15:15:00 | 1248.90 | 2024-12-30 09:15:00 | 1124.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-15 15:15:00 | 1009.40 | 2025-01-16 09:15:00 | 1065.85 | STOP_HIT | 1.00 | -5.59% |
| SELL | retest1 | 2025-02-04 10:45:00 | 894.75 | 2025-02-05 09:15:00 | 948.90 | STOP_HIT | 1.00 | -6.05% |
| SELL | retest1 | 2025-02-04 11:30:00 | 897.55 | 2025-02-05 09:15:00 | 948.90 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest1 | 2025-02-04 13:45:00 | 898.35 | 2025-02-05 09:15:00 | 948.90 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2025-02-05 11:45:00 | 939.05 | 2025-02-07 09:15:00 | 892.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 11:45:00 | 939.05 | 2025-02-10 10:15:00 | 895.60 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-02-27 09:15:00 | 760.80 | 2025-02-28 09:15:00 | 722.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 760.80 | 2025-03-03 09:15:00 | 684.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-03-13 09:15:00 | 689.95 | 2025-03-18 09:15:00 | 699.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-03-13 11:30:00 | 696.20 | 2025-03-18 14:15:00 | 710.25 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-03-13 12:15:00 | 696.60 | 2025-03-18 14:15:00 | 710.25 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-03-13 14:30:00 | 696.15 | 2025-03-18 14:15:00 | 710.25 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-03-17 11:00:00 | 696.85 | 2025-03-18 14:15:00 | 710.25 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-04-08 10:30:00 | 754.35 | 2025-04-15 09:15:00 | 783.00 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-04-09 09:15:00 | 745.60 | 2025-04-15 09:15:00 | 783.00 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2025-04-11 09:30:00 | 755.00 | 2025-04-15 09:15:00 | 783.00 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-04-11 12:15:00 | 754.00 | 2025-04-15 09:15:00 | 783.00 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-04-11 13:15:00 | 748.45 | 2025-04-15 09:15:00 | 783.00 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest2 | 2025-04-24 09:15:00 | 814.95 | 2025-04-25 10:15:00 | 774.60 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest2 | 2025-04-24 15:15:00 | 812.80 | 2025-04-25 10:15:00 | 774.60 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-05-08 12:45:00 | 714.60 | 2025-05-09 09:15:00 | 678.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 12:45:00 | 714.60 | 2025-05-09 15:15:00 | 691.00 | STOP_HIT | 0.50 | 3.30% |
| BUY | retest2 | 2025-05-21 09:45:00 | 920.40 | 2025-05-23 09:15:00 | 892.45 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-05-21 13:00:00 | 916.30 | 2025-05-23 09:15:00 | 892.45 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-06-02 11:15:00 | 898.00 | 2025-06-02 13:15:00 | 912.90 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-02 12:15:00 | 901.10 | 2025-06-02 13:15:00 | 912.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-03 10:45:00 | 899.90 | 2025-06-04 11:15:00 | 929.60 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-06-20 12:15:00 | 859.00 | 2025-06-23 09:15:00 | 883.30 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-06-20 14:00:00 | 860.00 | 2025-06-23 09:15:00 | 883.30 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-06-30 09:15:00 | 951.00 | 2025-07-03 12:15:00 | 943.60 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-17 15:00:00 | 947.10 | 2025-07-18 09:15:00 | 936.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-22 10:15:00 | 934.25 | 2025-07-25 11:15:00 | 887.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 934.25 | 2025-07-28 09:15:00 | 896.75 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-08-08 09:15:00 | 845.05 | 2025-08-08 14:15:00 | 802.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:15:00 | 845.05 | 2025-08-12 09:15:00 | 802.10 | STOP_HIT | 0.50 | 5.08% |
| BUY | retest2 | 2025-09-04 09:45:00 | 852.95 | 2025-09-04 10:15:00 | 844.55 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-09-16 14:15:00 | 940.85 | 2025-09-22 14:15:00 | 937.15 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-16 15:00:00 | 937.70 | 2025-09-22 14:15:00 | 937.15 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-09-26 09:15:00 | 889.35 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-26 11:15:00 | 890.25 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-09-29 10:00:00 | 892.10 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-09-29 10:30:00 | 893.40 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-10-06 13:15:00 | 887.40 | 2025-10-13 09:15:00 | 889.90 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-10-06 14:15:00 | 886.90 | 2025-10-13 09:15:00 | 889.90 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-10-15 13:45:00 | 888.05 | 2025-10-23 10:15:00 | 887.00 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-10-16 11:30:00 | 888.40 | 2025-10-23 10:15:00 | 887.00 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-10-23 09:45:00 | 886.20 | 2025-10-23 10:15:00 | 887.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-29 10:15:00 | 900.00 | 2025-10-31 13:15:00 | 894.75 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-19 13:30:00 | 872.80 | 2025-11-19 14:15:00 | 873.65 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-11-26 13:15:00 | 843.00 | 2025-11-26 14:15:00 | 848.05 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-27 11:15:00 | 842.80 | 2025-12-03 10:15:00 | 800.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:30:00 | 840.00 | 2025-12-03 10:15:00 | 798.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:15:00 | 842.80 | 2025-12-04 15:15:00 | 791.40 | STOP_HIT | 0.50 | 6.10% |
| SELL | retest2 | 2025-12-01 09:30:00 | 840.00 | 2025-12-04 15:15:00 | 791.40 | STOP_HIT | 0.50 | 5.79% |
| BUY | retest2 | 2025-12-30 10:30:00 | 922.70 | 2025-12-30 14:15:00 | 878.20 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2025-12-30 12:30:00 | 908.05 | 2025-12-30 14:15:00 | 878.20 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-01-02 13:15:00 | 886.20 | 2026-01-02 15:15:00 | 891.80 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-02 13:45:00 | 885.90 | 2026-01-02 15:15:00 | 891.80 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-05 09:15:00 | 884.00 | 2026-01-08 10:15:00 | 839.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:15:00 | 886.00 | 2026-01-08 10:15:00 | 841.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 881.65 | 2026-01-08 10:15:00 | 837.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:00:00 | 882.85 | 2026-01-08 10:15:00 | 838.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 882.50 | 2026-01-08 10:15:00 | 838.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 882.45 | 2026-01-08 10:15:00 | 838.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 12:15:00 | 865.00 | 2026-01-08 15:15:00 | 821.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 884.00 | 2026-01-12 09:15:00 | 795.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 10:15:00 | 886.00 | 2026-01-12 09:15:00 | 797.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 881.65 | 2026-01-12 09:15:00 | 793.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 13:00:00 | 882.85 | 2026-01-12 09:15:00 | 794.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 882.50 | 2026-01-12 09:15:00 | 794.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 882.45 | 2026-01-12 09:15:00 | 794.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 12:15:00 | 865.00 | 2026-01-12 11:15:00 | 778.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 781.60 | 2026-01-22 14:15:00 | 796.60 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-01-22 12:30:00 | 782.90 | 2026-01-22 14:15:00 | 796.60 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-30 11:00:00 | 817.05 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-01-30 12:45:00 | 816.90 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2026-01-30 13:30:00 | 820.65 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-01-30 14:45:00 | 817.25 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2026-02-04 09:15:00 | 790.25 | 2026-02-10 11:15:00 | 788.45 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2026-02-04 10:15:00 | 792.55 | 2026-02-10 11:15:00 | 788.45 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-02-04 15:00:00 | 794.00 | 2026-02-10 11:15:00 | 788.45 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-02-19 09:15:00 | 759.85 | 2026-02-25 13:15:00 | 721.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 759.85 | 2026-03-02 09:15:00 | 683.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 15:00:00 | 656.00 | 2026-03-16 09:15:00 | 623.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 651.20 | 2026-03-16 09:15:00 | 618.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 15:00:00 | 656.00 | 2026-03-16 14:15:00 | 629.95 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2026-03-13 09:15:00 | 651.20 | 2026-03-16 14:15:00 | 629.95 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2026-03-20 12:15:00 | 629.40 | 2026-03-23 10:15:00 | 597.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 629.20 | 2026-03-23 10:15:00 | 597.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 628.65 | 2026-03-23 12:15:00 | 597.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 629.40 | 2026-03-24 10:15:00 | 606.00 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2026-03-20 13:30:00 | 629.20 | 2026-03-24 10:15:00 | 606.00 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-03-20 14:30:00 | 628.65 | 2026-03-24 10:15:00 | 606.00 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2026-04-13 10:45:00 | 699.45 | 2026-04-13 14:15:00 | 697.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-04-13 11:15:00 | 699.25 | 2026-04-13 14:15:00 | 697.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-13 13:45:00 | 700.85 | 2026-04-13 14:15:00 | 697.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-04-23 09:15:00 | 747.15 | 2026-04-24 10:15:00 | 727.25 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-04-23 09:45:00 | 746.50 | 2026-04-24 10:15:00 | 727.25 | STOP_HIT | 1.00 | -2.58% |
