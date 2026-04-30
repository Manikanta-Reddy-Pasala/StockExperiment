# Torrent Power Ltd. (TORNTPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1736.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 326.22
- **Avg P&L per closed trade:** 46.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 1591.55 | 1777.56 | 1777.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1579.15 | 1764.53 | 1771.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 14:15:00 | 1675.90 | 1670.99 | 1714.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 11:15:00 | 1653.95 | 1670.81 | 1711.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 1685.50 | 1664.01 | 1701.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-13 09:15:00 | 1670.00 | 1664.31 | 1701.37 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 1401.70 | 1327.58 | 1384.88 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 11:15:00 | 1515.05 | 1422.31 | 1422.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 12:15:00 | 1537.10 | 1423.45 | 1422.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 1508.70 | 1518.72 | 1483.16 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 1396.10 | 1464.89 | 1464.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1378.70 | 1446.57 | 1454.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 13:15:00 | 1441.50 | 1431.22 | 1444.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-11 13:15:00 | 1424.50 | 1432.50 | 1444.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1424.50 | 1432.50 | 1444.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-12 10:15:00 | 1421.70 | 1432.40 | 1444.13 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-24 10:15:00 | 1442.00 | 1419.39 | 1433.94 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 1320.00 | 1300.31 | 1300.27 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1259.10 | 1300.55 | 1300.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1252.00 | 1300.07 | 1300.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1293.00 | 1292.67 | 1296.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 11:15:00 | 1280.80 | 1292.23 | 1295.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1295.60 | 1292.17 | 1295.74 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-17 14:15:00 | 1275.00 | 1291.62 | 1295.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1288.00 | 1287.53 | 1292.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 11:15:00 | 1295.20 | 1287.65 | 1292.90 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 1400.60 | 1296.27 | 1295.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 1402.70 | 1299.37 | 1297.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 13:15:00 | 1320.90 | 1328.92 | 1316.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-28 10:15:00 | 1332.90 | 1322.13 | 1314.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1327.30 | 1328.28 | 1318.23 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-02 10:15:00 | 1307.80 | 1328.08 | 1318.18 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-13 09:15:00 | 1670.00 | 2024-12-19 09:15:00 | 1575.89 | TARGET | 94.11 |
| SELL | 2024-12-06 11:15:00 | 1653.95 | 2024-12-20 14:15:00 | 1482.53 | TARGET | 171.42 |
| SELL | 2025-06-11 13:15:00 | 1424.50 | 2025-06-19 12:15:00 | 1364.76 | TARGET | 59.74 |
| SELL | 2025-06-12 10:15:00 | 1421.70 | 2025-06-24 10:15:00 | 1442.00 | EXIT_EMA400 | -20.30 |
| SELL | 2025-12-16 11:15:00 | 1280.80 | 2025-12-22 11:15:00 | 1295.20 | EXIT_EMA400 | -14.40 |
| SELL | 2025-12-17 14:15:00 | 1275.00 | 2025-12-22 11:15:00 | 1295.20 | EXIT_EMA400 | -20.20 |
| BUY | 2026-01-28 10:15:00 | 1332.90 | 2026-01-30 14:15:00 | 1388.74 | TARGET | 55.84 |
