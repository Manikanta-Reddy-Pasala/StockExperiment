# CreditAccess Grameen Ltd. (CREDITACC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1300.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -263.91
- **Avg P&L per closed trade:** -32.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 13:15:00 | 1310.70 | 1356.40 | 1356.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 1301.80 | 1355.08 | 1355.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 1355.30 | 1351.56 | 1354.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-11 10:15:00 | 1346.50 | 1351.51 | 1354.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 1346.50 | 1351.51 | 1354.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-10-11 11:15:00 | 1355.00 | 1351.55 | 1354.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 14:15:00 | 1395.55 | 1356.39 | 1356.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-23 09:15:00 | 1500.00 | 1367.22 | 1362.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 1678.75 | 1684.40 | 1608.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-20 12:15:00 | 1702.45 | 1684.54 | 1609.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 1627.45 | 1683.42 | 1610.89 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-22 11:15:00 | 1679.95 | 1679.03 | 1611.83 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-12-26 09:15:00 | 1607.65 | 1676.54 | 1612.23 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 10:15:00 | 1568.65 | 1616.52 | 1616.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 11:15:00 | 1567.65 | 1616.03 | 1616.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 1449.80 | 1436.82 | 1491.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-01 10:15:00 | 1424.00 | 1436.69 | 1490.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1478.65 | 1432.53 | 1483.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-04 10:15:00 | 1492.30 | 1433.13 | 1483.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 1050.60 | 963.25 | 962.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 1072.65 | 965.24 | 963.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 966.65 | 986.74 | 975.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-12 11:15:00 | 1004.30 | 987.03 | 976.11 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-14 12:15:00 | 976.90 | 989.89 | 978.39 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 863.70 | 969.56 | 969.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 851.30 | 966.30 | 968.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 974.50 | 957.53 | 963.43 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 13:15:00 | 981.80 | 958.99 | 958.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 14:15:00 | 1002.10 | 959.42 | 959.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 941.50 | 959.63 | 959.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 963.00 | 959.31 | 959.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 963.00 | 959.31 | 959.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 09:15:00 | 967.55 | 959.39 | 959.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 1119.60 | 1158.81 | 1120.09 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1255.30 | 1350.58 | 1350.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1252.70 | 1349.61 | 1350.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 1313.10 | 1311.00 | 1326.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 13:15:00 | 1288.10 | 1310.10 | 1325.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 11:15:00 | 1317.50 | 1297.85 | 1315.79 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-11 10:15:00 | 1346.50 | 2023-10-11 11:15:00 | 1355.00 | EXIT_EMA400 | -8.50 |
| BUY | 2023-12-20 12:15:00 | 1702.45 | 2023-12-26 09:15:00 | 1607.65 | EXIT_EMA400 | -94.80 |
| BUY | 2023-12-22 11:15:00 | 1679.95 | 2023-12-26 09:15:00 | 1607.65 | EXIT_EMA400 | -72.30 |
| SELL | 2024-04-01 10:15:00 | 1424.00 | 2024-04-04 10:15:00 | 1492.30 | EXIT_EMA400 | -68.30 |
| BUY | 2025-02-12 11:15:00 | 1004.30 | 2025-02-14 12:15:00 | 976.90 | EXIT_EMA400 | -27.40 |
| BUY | 2025-04-07 15:15:00 | 963.00 | 2025-04-08 09:15:00 | 974.63 | TARGET | 11.63 |
| BUY | 2025-04-08 09:15:00 | 967.55 | 2025-04-08 12:15:00 | 992.71 | TARGET | 25.16 |
| SELL | 2025-12-26 13:15:00 | 1288.10 | 2026-01-05 11:15:00 | 1317.50 | EXIT_EMA400 | -29.40 |
