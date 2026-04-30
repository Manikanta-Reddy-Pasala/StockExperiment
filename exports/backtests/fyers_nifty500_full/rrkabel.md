# R R Kabel Ltd. (RRKABEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1571.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 457.49
- **Avg P&L per closed trade:** 76.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 1626.40 | 1728.60 | 1728.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 1621.95 | 1725.53 | 1727.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 14:15:00 | 1715.15 | 1630.59 | 1664.32 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 09:15:00 | 1717.45 | 1678.76 | 1678.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-07 14:15:00 | 1772.80 | 1680.89 | 1679.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 09:15:00 | 1699.75 | 1711.28 | 1697.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-21 09:15:00 | 1726.10 | 1711.76 | 1698.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-21 14:15:00 | 1685.05 | 1711.20 | 1698.09 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 1527.85 | 1686.91 | 1686.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 14:15:00 | 1513.10 | 1685.18 | 1686.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 15:15:00 | 1328.00 | 1316.62 | 1387.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 1264.40 | 1316.10 | 1387.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 990.60 | 955.77 | 1019.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-30 10:15:00 | 1043.00 | 956.63 | 1020.10 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 12:15:00 | 1318.00 | 1065.74 | 1064.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1321.70 | 1091.14 | 1078.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1310.00 | 1310.04 | 1238.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 12:15:00 | 1350.50 | 1311.53 | 1244.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 14:15:00 | 1334.60 | 1389.49 | 1340.74 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 1215.60 | 1308.86 | 1309.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1204.70 | 1284.45 | 1295.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 1243.00 | 1242.64 | 1266.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-17 13:15:00 | 1239.00 | 1244.45 | 1264.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1256.80 | 1244.44 | 1263.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-22 09:15:00 | 1272.70 | 1245.36 | 1263.17 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 1392.00 | 1266.07 | 1266.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 1401.70 | 1267.42 | 1266.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1339.00 | 1341.13 | 1313.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-19 11:15:00 | 1363.70 | 1341.36 | 1314.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1321.60 | 1344.88 | 1318.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-24 11:15:00 | 1328.90 | 1344.73 | 1318.67 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-19 11:15:00 | 1422.60 | 1470.56 | 1428.13 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1318.60 | 1432.18 | 1432.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1306.00 | 1413.82 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1397.40 | 1385.43 | 1405.51 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1505.60 | 1416.39 | 1416.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 1510.10 | 1417.33 | 1416.57 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-21 09:15:00 | 1726.10 | 2024-10-21 14:15:00 | 1685.05 | EXIT_EMA400 | -41.05 |
| SELL | 2025-02-03 09:15:00 | 1264.40 | 2025-02-27 14:15:00 | 895.60 | TARGET | 368.80 |
| BUY | 2025-06-23 12:15:00 | 1350.50 | 2025-08-01 14:15:00 | 1334.60 | EXIT_EMA400 | -15.90 |
| SELL | 2025-09-17 13:15:00 | 1239.00 | 2025-09-22 09:15:00 | 1272.70 | EXIT_EMA400 | -33.70 |
| BUY | 2025-11-24 11:15:00 | 1328.90 | 2025-11-25 13:15:00 | 1359.59 | TARGET | 30.69 |
| BUY | 2025-11-19 11:15:00 | 1363.70 | 2025-12-22 09:15:00 | 1512.34 | TARGET | 148.64 |
