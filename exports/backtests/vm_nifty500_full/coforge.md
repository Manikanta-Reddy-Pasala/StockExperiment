# Coforge Ltd. (COFORGE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1195.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 251.67
- **Avg P&L per closed trade:** 27.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 1005.96 | 1019.08 | 1019.09 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 12:15:00 | 1021.97 | 1018.94 | 1018.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 13:15:00 | 1024.19 | 1018.99 | 1018.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 09:15:00 | 1012.02 | 1019.10 | 1019.01 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 11:15:00 | 1008.20 | 1018.90 | 1018.91 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1054.19 | 1019.02 | 1018.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 12:15:00 | 1059.59 | 1020.08 | 1019.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 09:15:00 | 1179.93 | 1192.36 | 1140.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-05 09:15:00 | 1218.70 | 1192.09 | 1144.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-05 13:15:00 | 1259.00 | 1295.12 | 1259.97 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 12:15:00 | 1139.80 | 1239.30 | 1239.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 09:15:00 | 1110.20 | 1235.15 | 1237.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 1016.00 | 990.87 | 1060.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 11:15:00 | 978.08 | 1001.74 | 1049.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1043.27 | 1003.43 | 1047.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-07 09:15:00 | 1087.00 | 1006.47 | 1047.37 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 1173.94 | 1063.04 | 1062.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 12:15:00 | 1177.99 | 1065.28 | 1063.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1157.73 | 1186.49 | 1142.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 09:15:00 | 1211.21 | 1185.77 | 1143.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-13 14:15:00 | 1740.07 | 1844.13 | 1741.97 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 1543.89 | 1713.99 | 1714.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 1513.89 | 1705.74 | 1710.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 11:15:00 | 1559.00 | 1546.26 | 1599.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-21 13:15:00 | 1540.63 | 1546.28 | 1598.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1578.60 | 1547.32 | 1596.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 11:15:00 | 1601.60 | 1548.16 | 1596.50 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1677.60 | 1532.09 | 1531.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1696.10 | 1540.98 | 1536.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1845.20 | 1856.63 | 1780.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-18 11:15:00 | 1874.50 | 1856.75 | 1781.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-24 09:15:00 | 1698.60 | 1855.84 | 1790.24 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1638.20 | 1752.58 | 1752.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 1631.00 | 1751.37 | 1752.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 1719.80 | 1719.14 | 1734.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 09:15:00 | 1657.30 | 1728.79 | 1735.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1754.20 | 1719.70 | 1729.95 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 1812.20 | 1738.59 | 1738.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 1816.00 | 1740.09 | 1739.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1733.20 | 1745.76 | 1742.00 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 1636.10 | 1737.64 | 1738.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1629.70 | 1736.56 | 1737.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1686.00 | 1678.10 | 1703.27 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1832.10 | 1717.89 | 1717.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 1839.80 | 1765.91 | 1747.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 1847.10 | 1852.40 | 1807.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-15 10:15:00 | 1865.50 | 1850.23 | 1810.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-23 09:15:00 | 1798.00 | 1851.72 | 1818.87 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 1661.90 | 1793.29 | 1793.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 14:15:00 | 1660.20 | 1790.67 | 1792.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 1739.50 | 1736.02 | 1758.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 1708.90 | 1735.40 | 1757.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1706.70 | 1703.25 | 1733.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 09:15:00 | 1607.90 | 1703.13 | 1732.05 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1308.20 | 1220.29 | 1317.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 13:15:00 | 1319.60 | 1223.96 | 1317.49 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-05 09:15:00 | 1218.70 | 2024-03-05 13:15:00 | 1259.00 | EXIT_EMA400 | 40.30 |
| SELL | 2024-06-04 11:15:00 | 978.08 | 2024-06-07 09:15:00 | 1087.00 | EXIT_EMA400 | -108.92 |
| BUY | 2024-08-06 09:15:00 | 1211.21 | 2024-09-16 11:15:00 | 1413.98 | TARGET | 202.77 |
| SELL | 2025-03-21 13:15:00 | 1540.63 | 2025-03-25 11:15:00 | 1601.60 | EXIT_EMA400 | -60.97 |
| BUY | 2025-07-18 11:15:00 | 1874.50 | 2025-07-24 09:15:00 | 1698.60 | EXIT_EMA400 | -175.90 |
| SELL | 2025-09-05 09:15:00 | 1657.30 | 2025-09-10 09:15:00 | 1754.20 | EXIT_EMA400 | -96.90 |
| BUY | 2025-12-15 10:15:00 | 1865.50 | 2025-12-23 09:15:00 | 1798.00 | EXIT_EMA400 | -67.50 |
| SELL | 2026-01-20 09:15:00 | 1708.90 | 2026-02-06 09:15:00 | 1562.55 | TARGET | 146.35 |
| SELL | 2026-02-04 09:15:00 | 1607.90 | 2026-02-24 09:15:00 | 1235.46 | TARGET | 372.44 |
