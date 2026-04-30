# Astral Ltd. (ASTRAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1533.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -82.00
- **Avg P&L per closed trade:** -16.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 2095.35 | 2228.03 | 2228.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 2085.85 | 2226.61 | 2227.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 1971.00 | 1958.95 | 2027.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-03 12:15:00 | 1930.25 | 1972.74 | 2020.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 13:15:00 | 1851.95 | 1797.19 | 1848.96 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 1511.80 | 1377.05 | 1376.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 1528.50 | 1382.68 | 1379.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1486.30 | 1493.86 | 1459.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 14:15:00 | 1507.40 | 1490.86 | 1467.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1478.30 | 1493.85 | 1473.14 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 09:15:00 | 1459.70 | 1493.31 | 1473.07 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1409.50 | 1459.90 | 1459.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 1401.50 | 1455.96 | 1457.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1438.60 | 1402.69 | 1426.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 10:15:00 | 1390.60 | 1406.10 | 1426.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-02 11:15:00 | 1426.40 | 1400.15 | 1420.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 1451.50 | 1431.35 | 1431.34 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1404.00 | 1431.45 | 1431.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1400.50 | 1431.15 | 1431.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 1417.50 | 1411.74 | 1420.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-09 09:15:00 | 1399.10 | 1411.28 | 1419.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 11:15:00 | 1422.40 | 1410.91 | 1419.18 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1453.00 | 1424.47 | 1424.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 1457.30 | 1425.97 | 1425.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1468.50 | 1482.73 | 1458.44 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 1405.70 | 1449.37 | 1449.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 1395.90 | 1436.24 | 1442.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 13:15:00 | 1426.70 | 1420.72 | 1432.67 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 1472.60 | 1441.94 | 1441.82 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 1416.50 | 1441.52 | 1441.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1404.10 | 1441.15 | 1441.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 1428.80 | 1426.65 | 1433.50 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 1506.40 | 1439.50 | 1439.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 1510.80 | 1453.65 | 1446.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 15:15:00 | 1610.00 | 1610.66 | 1556.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 09:15:00 | 1617.00 | 1610.72 | 1556.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 1563.50 | 1618.74 | 1569.63 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-03 12:15:00 | 1930.25 | 2024-12-03 13:15:00 | 1851.95 | EXIT_EMA400 | 78.30 |
| BUY | 2025-07-15 14:15:00 | 1507.40 | 2025-07-23 09:15:00 | 1459.70 | EXIT_EMA400 | -47.70 |
| SELL | 2025-08-26 10:15:00 | 1390.60 | 2025-09-02 11:15:00 | 1426.40 | EXIT_EMA400 | -35.80 |
| SELL | 2025-10-09 09:15:00 | 1399.10 | 2025-10-10 11:15:00 | 1422.40 | EXIT_EMA400 | -23.30 |
| BUY | 2026-03-16 09:15:00 | 1617.00 | 2026-03-23 09:15:00 | 1563.50 | EXIT_EMA400 | -53.50 |
