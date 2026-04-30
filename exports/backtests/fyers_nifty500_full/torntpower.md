# Torrent Power Ltd. (TORNTPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1738.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 5 |
| EXIT | 4 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 398.96
- **Avg P&L per closed trade:** 44.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 1588.00 | 1777.22 | 1777.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 1568.15 | 1762.28 | 1770.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 14:15:00 | 1675.90 | 1670.83 | 1714.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 09:15:00 | 1658.90 | 1671.15 | 1714.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1658.90 | 1671.15 | 1714.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-06 12:15:00 | 1641.05 | 1670.57 | 1710.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 1685.50 | 1663.96 | 1701.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-12 15:15:00 | 1680.00 | 1664.12 | 1701.42 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 1401.70 | 1327.07 | 1383.52 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 10:15:00 | 1533.70 | 1421.03 | 1420.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 12:15:00 | 1537.10 | 1423.11 | 1421.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 1508.70 | 1518.51 | 1482.55 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1396.00 | 1464.20 | 1464.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 1388.10 | 1451.19 | 1457.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 13:15:00 | 1441.50 | 1431.25 | 1444.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-11 13:15:00 | 1424.50 | 1432.58 | 1444.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1424.50 | 1432.58 | 1444.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-12 10:15:00 | 1421.70 | 1432.50 | 1443.98 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-24 10:15:00 | 1442.00 | 1419.42 | 1433.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 1320.00 | 1300.28 | 1300.25 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1259.10 | 1300.52 | 1300.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1252.00 | 1300.04 | 1300.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1293.00 | 1292.59 | 1296.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 11:15:00 | 1280.80 | 1292.16 | 1295.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1295.60 | 1292.10 | 1295.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-17 14:15:00 | 1275.00 | 1291.54 | 1295.27 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1288.20 | 1287.49 | 1292.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 11:15:00 | 1295.20 | 1287.61 | 1292.88 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 1400.60 | 1296.22 | 1295.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 1402.70 | 1299.32 | 1297.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 15:15:00 | 1320.10 | 1320.53 | 1309.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-12 09:15:00 | 1329.10 | 1320.62 | 1309.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1329.10 | 1320.62 | 1309.48 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-12 11:15:00 | 1343.60 | 1320.95 | 1309.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-20 15:15:00 | 1313.50 | 1328.63 | 1315.99 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-12 15:15:00 | 1680.00 | 2024-12-17 12:15:00 | 1615.73 | TARGET | 64.27 |
| SELL | 2024-12-04 09:15:00 | 1658.90 | 2024-12-20 14:15:00 | 1491.91 | TARGET | 166.99 |
| SELL | 2024-12-06 12:15:00 | 1641.05 | 2024-12-30 09:15:00 | 1432.01 | TARGET | 209.04 |
| SELL | 2025-06-11 13:15:00 | 1424.50 | 2025-06-19 12:15:00 | 1365.24 | TARGET | 59.26 |
| SELL | 2025-06-12 10:15:00 | 1421.70 | 2025-06-24 10:15:00 | 1442.00 | EXIT_EMA400 | -20.30 |
| SELL | 2025-12-16 11:15:00 | 1280.80 | 2025-12-22 11:15:00 | 1295.20 | EXIT_EMA400 | -14.40 |
| SELL | 2025-12-17 14:15:00 | 1275.00 | 2025-12-22 11:15:00 | 1295.20 | EXIT_EMA400 | -20.20 |
| BUY | 2026-01-12 09:15:00 | 1329.10 | 2026-01-20 15:15:00 | 1313.50 | EXIT_EMA400 | -15.60 |
| BUY | 2026-01-12 11:15:00 | 1343.60 | 2026-01-20 15:15:00 | 1313.50 | EXIT_EMA400 | -30.10 |
