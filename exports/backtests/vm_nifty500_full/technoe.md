# Techno Electric & Engineering Company Ltd. (TECHNOE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1283.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 21.02
- **Avg P&L per closed trade:** 5.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 13:15:00 | 643.80 | 741.40 | 741.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 638.60 | 726.41 | 733.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 737.50 | 709.57 | 723.78 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 09:15:00 | 817.25 | 734.56 | 734.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 10:15:00 | 830.95 | 740.53 | 737.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 10:15:00 | 1013.70 | 1022.63 | 938.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-29 12:15:00 | 1050.05 | 1023.12 | 939.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1551.00 | 1627.72 | 1545.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-09 09:15:00 | 1599.65 | 1626.72 | 1545.93 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-19 11:15:00 | 1538.00 | 1613.59 | 1559.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 13:15:00 | 1439.70 | 1583.05 | 1583.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 1416.50 | 1507.62 | 1534.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 11:15:00 | 1506.95 | 1497.84 | 1527.40 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 1692.25 | 1545.20 | 1544.79 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 15:15:00 | 1402.45 | 1545.63 | 1546.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1387.80 | 1544.06 | 1545.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 994.10 | 992.24 | 1112.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 909.25 | 1004.50 | 1078.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1034.40 | 994.90 | 1058.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 11:15:00 | 1078.90 | 996.34 | 1059.02 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 1263.90 | 1085.73 | 1084.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 15:15:00 | 1283.90 | 1091.31 | 1087.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1525.10 | 1526.06 | 1435.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-22 09:15:00 | 1547.00 | 1462.13 | 1439.51 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1471.00 | 1498.72 | 1470.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-10 12:15:00 | 1468.00 | 1498.42 | 1470.09 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1309.90 | 1454.26 | 1454.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1298.10 | 1452.71 | 1453.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 1014.00 | 1008.03 | 1086.07 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1219.00 | 1093.80 | 1093.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 13:15:00 | 1233.25 | 1098.99 | 1096.17 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-29 12:15:00 | 1050.05 | 2024-05-31 14:15:00 | 1381.37 | TARGET | 331.32 |
| BUY | 2024-09-09 09:15:00 | 1599.65 | 2024-09-19 11:15:00 | 1538.00 | EXIT_EMA400 | -61.65 |
| SELL | 2025-04-07 09:15:00 | 909.25 | 2025-04-17 11:15:00 | 1078.90 | EXIT_EMA400 | -169.65 |
| BUY | 2025-08-22 09:15:00 | 1547.00 | 2025-09-10 12:15:00 | 1468.00 | EXIT_EMA400 | -79.00 |
