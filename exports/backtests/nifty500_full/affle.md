# Affle 3i Ltd. (AFFLE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1421.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 244.33
- **Avg P&L per closed trade:** 24.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 13:15:00 | 1025.10 | 1079.36 | 1079.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 1008.50 | 1077.59 | 1078.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 14:15:00 | 1068.85 | 1067.67 | 1072.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-08 13:15:00 | 1052.70 | 1067.60 | 1072.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 1059.10 | 1053.41 | 1063.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-21 11:15:00 | 1057.45 | 1053.45 | 1062.99 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1059.60 | 1053.58 | 1062.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-22 12:15:00 | 1063.00 | 1053.85 | 1062.82 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 1116.05 | 1069.97 | 1069.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 1126.80 | 1070.98 | 1070.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 1249.20 | 1249.77 | 1197.05 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 1136.00 | 1186.30 | 1186.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 12:15:00 | 1130.50 | 1185.75 | 1186.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 1096.45 | 1085.96 | 1118.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-09 13:15:00 | 1076.30 | 1088.99 | 1114.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 1098.30 | 1086.85 | 1109.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-18 12:15:00 | 1095.00 | 1086.93 | 1109.28 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 1101.35 | 1081.75 | 1102.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-26 13:15:00 | 1102.50 | 1082.15 | 1102.26 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 1196.00 | 1108.67 | 1108.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 1225.00 | 1115.32 | 1111.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 1135.05 | 1141.62 | 1127.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-03 10:15:00 | 1148.85 | 1141.67 | 1127.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 09:15:00 | 1119.00 | 1142.00 | 1128.24 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 11:15:00 | 1558.25 | 1651.91 | 1652.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 1537.50 | 1647.30 | 1649.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1588.80 | 1583.66 | 1612.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 1514.85 | 1596.46 | 1615.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-14 10:15:00 | 1611.85 | 1590.03 | 1610.62 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 1622.00 | 1543.31 | 1543.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1623.90 | 1545.60 | 1544.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 1558.00 | 1558.20 | 1551.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 11:15:00 | 1593.50 | 1550.83 | 1548.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 1843.60 | 1914.77 | 1848.43 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 1916.20 | 1949.92 | 1949.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 1890.00 | 1949.32 | 1949.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1954.70 | 1945.48 | 1947.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 10:15:00 | 1928.00 | 1945.31 | 1947.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1928.00 | 1945.31 | 1947.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-31 15:15:00 | 1925.30 | 1944.75 | 1947.20 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1898.00 | 1944.29 | 1946.96 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-03 15:15:00 | 1892.00 | 1941.91 | 1945.68 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1761.50 | 1706.06 | 1764.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-23 12:15:00 | 1772.10 | 1706.72 | 1764.87 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-08 13:15:00 | 1052.70 | 2023-11-22 12:15:00 | 1063.00 | EXIT_EMA400 | -10.30 |
| SELL | 2023-11-21 11:15:00 | 1057.45 | 2023-11-22 12:15:00 | 1063.00 | EXIT_EMA400 | -5.55 |
| SELL | 2024-04-09 13:15:00 | 1076.30 | 2024-04-26 13:15:00 | 1102.50 | EXIT_EMA400 | -26.20 |
| SELL | 2024-04-18 12:15:00 | 1095.00 | 2024-04-26 13:15:00 | 1102.50 | EXIT_EMA400 | -7.50 |
| BUY | 2024-06-03 10:15:00 | 1148.85 | 2024-06-04 09:15:00 | 1119.00 | EXIT_EMA400 | -29.85 |
| SELL | 2025-02-12 09:15:00 | 1514.85 | 2025-02-14 10:15:00 | 1611.85 | EXIT_EMA400 | -97.00 |
| BUY | 2025-05-12 11:15:00 | 1593.50 | 2025-05-16 11:15:00 | 1728.88 | TARGET | 135.38 |
| SELL | 2025-10-31 10:15:00 | 1928.00 | 2025-11-04 09:15:00 | 1869.38 | TARGET | 58.62 |
| SELL | 2025-10-31 15:15:00 | 1925.30 | 2025-11-04 09:15:00 | 1859.59 | TARGET | 65.71 |
| SELL | 2025-11-03 15:15:00 | 1892.00 | 2025-11-10 09:15:00 | 1730.97 | TARGET | 161.03 |
