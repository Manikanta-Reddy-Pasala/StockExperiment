# Olectra Greentech Ltd. (OLECTRA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1245.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 101.96
- **Avg P&L per closed trade:** 14.57

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 1633.05 | 1767.06 | 1767.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 1626.70 | 1764.38 | 1765.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 1769.00 | 1724.72 | 1742.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 10:15:00 | 1663.00 | 1754.22 | 1755.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-07 12:15:00 | 1749.85 | 1736.51 | 1745.43 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1817.00 | 1751.54 | 1751.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 10:15:00 | 1831.00 | 1752.33 | 1751.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 1802.10 | 1803.56 | 1782.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-16 09:15:00 | 1831.95 | 1804.44 | 1784.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1787.00 | 1804.91 | 1785.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-18 10:15:00 | 1769.25 | 1804.55 | 1784.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 1731.90 | 1770.93 | 1770.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 13:15:00 | 1723.00 | 1770.45 | 1770.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 1723.90 | 1692.83 | 1726.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-16 10:15:00 | 1618.20 | 1687.25 | 1721.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1616.25 | 1602.40 | 1651.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-17 11:15:00 | 1670.80 | 1607.75 | 1648.17 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 1716.55 | 1665.27 | 1665.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 1724.00 | 1665.86 | 1665.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 09:15:00 | 1662.45 | 1670.59 | 1668.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-18 11:15:00 | 1676.05 | 1670.65 | 1668.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1693.00 | 1674.34 | 1670.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-22 12:15:00 | 1667.45 | 1674.28 | 1670.21 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 1600.00 | 1666.13 | 1666.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1582.50 | 1646.13 | 1654.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 1559.95 | 1551.07 | 1596.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 1537.80 | 1574.40 | 1592.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-30 10:15:00 | 1477.20 | 1413.07 | 1469.34 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1336.40 | 1222.06 | 1221.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 15:15:00 | 1348.00 | 1225.59 | 1223.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1212.50 | 1228.82 | 1225.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 09:15:00 | 1252.90 | 1227.42 | 1224.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1228.60 | 1229.00 | 1225.75 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-05 14:15:00 | 1219.80 | 1229.29 | 1226.09 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1172.30 | 1223.80 | 1223.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1158.00 | 1213.83 | 1218.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.13 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 1278.30 | 1213.36 | 1213.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 1284.20 | 1217.33 | 1215.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 1573.20 | 1578.39 | 1495.72 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 1412.40 | 1486.53 | 1486.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1391.10 | 1481.15 | 1484.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 1261.00 | 1257.09 | 1329.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 10:15:00 | 1209.00 | 1254.51 | 1323.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-18 14:15:00 | 1060.20 | 975.23 | 1049.40 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 1227.75 | 1073.46 | 1073.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 11:15:00 | 1233.60 | 1094.06 | 1083.92 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-04 10:15:00 | 1663.00 | 2024-06-07 12:15:00 | 1749.85 | EXIT_EMA400 | -86.85 |
| BUY | 2024-07-16 09:15:00 | 1831.95 | 2024-07-18 10:15:00 | 1769.25 | EXIT_EMA400 | -62.70 |
| SELL | 2024-08-16 10:15:00 | 1618.20 | 2024-09-17 11:15:00 | 1670.80 | EXIT_EMA400 | -52.60 |
| BUY | 2024-10-18 11:15:00 | 1676.05 | 2024-10-18 13:15:00 | 1699.78 | TARGET | 23.73 |
| SELL | 2024-12-18 09:15:00 | 1537.80 | 2025-01-10 09:15:00 | 1373.11 | TARGET | 164.69 |
| BUY | 2025-06-02 09:15:00 | 1252.90 | 2025-06-05 14:15:00 | 1219.80 | EXIT_EMA400 | -33.10 |
| SELL | 2025-12-29 10:15:00 | 1209.00 | 2026-03-18 14:15:00 | 1060.20 | EXIT_EMA400 | 148.80 |
