# Max Financial Services Ltd. (MFSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1585.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 1 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -194.10
- **Avg P&L per closed trade:** -32.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 12:15:00 | 907.90 | 944.80 | 944.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 878.90 | 943.02 | 944.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 13:15:00 | 916.80 | 915.51 | 927.54 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 949.20 | 936.93 | 936.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 977.30 | 938.05 | 937.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-22 10:15:00 | 944.00 | 944.57 | 941.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-22 14:15:00 | 950.60 | 944.64 | 941.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-23 09:15:00 | 941.00 | 944.66 | 941.21 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 926.70 | 983.52 | 983.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 912.00 | 982.81 | 983.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-11 12:15:00 | 963.35 | 962.95 | 971.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-11 14:15:00 | 955.00 | 962.86 | 971.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-13 09:15:00 | 990.00 | 962.87 | 971.27 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 1003.40 | 976.76 | 976.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 1017.60 | 982.01 | 979.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 1057.80 | 1059.03 | 1032.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-23 11:15:00 | 1069.15 | 1048.57 | 1031.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-29 13:15:00 | 1036.90 | 1053.82 | 1036.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 12:15:00 | 1143.80 | 1168.50 | 1168.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1129.45 | 1167.36 | 1167.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 14:15:00 | 1085.75 | 1083.02 | 1109.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 1066.10 | 1091.65 | 1108.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-11 09:15:00 | 1074.95 | 1048.30 | 1072.94 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 1154.95 | 1085.66 | 1085.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 10:15:00 | 1164.00 | 1098.99 | 1092.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 13:15:00 | 1558.30 | 1558.53 | 1472.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-11 14:15:00 | 1569.50 | 1558.64 | 1473.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-30 09:15:00 | 1496.90 | 1548.80 | 1497.75 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1550.00 | 1561.86 | 1561.89 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 1583.10 | 1561.98 | 1561.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 1587.60 | 1562.42 | 1562.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 1672.20 | 1673.55 | 1641.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-18 11:15:00 | 1686.00 | 1672.55 | 1642.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1650.80 | 1675.30 | 1650.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-29 15:15:00 | 1650.20 | 1675.05 | 1650.37 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 1596.70 | 1649.49 | 1649.49 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 1688.20 | 1649.55 | 1649.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 12:15:00 | 1702.00 | 1650.07 | 1649.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1753.20 | 1772.57 | 1727.26 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1572.50 | 1704.37 | 1704.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 10:15:00 | 1550.00 | 1701.55 | 1703.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 1627.50 | 1619.12 | 1654.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 11:15:00 | 1597.90 | 1636.92 | 1656.35 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-22 14:15:00 | 950.60 | 2024-02-23 09:15:00 | 941.00 | EXIT_EMA400 | -9.60 |
| SELL | 2024-06-11 14:15:00 | 955.00 | 2024-06-13 09:15:00 | 990.00 | EXIT_EMA400 | -35.00 |
| BUY | 2024-08-23 11:15:00 | 1069.15 | 2024-08-29 13:15:00 | 1036.90 | EXIT_EMA400 | -32.25 |
| SELL | 2025-02-11 09:15:00 | 1066.10 | 2025-03-11 09:15:00 | 1074.95 | EXIT_EMA400 | -8.85 |
| BUY | 2025-07-11 14:15:00 | 1569.50 | 2025-07-30 09:15:00 | 1496.90 | EXIT_EMA400 | -72.60 |
| BUY | 2025-12-18 11:15:00 | 1686.00 | 2025-12-29 15:15:00 | 1650.20 | EXIT_EMA400 | -35.80 |
