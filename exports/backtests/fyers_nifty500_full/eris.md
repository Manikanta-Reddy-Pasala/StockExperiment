# Eris Lifesciences Ltd. (ERIS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1321.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 386.95
- **Avg P&L per closed trade:** 77.39

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 1284.40 | 1360.57 | 1360.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1253.85 | 1356.63 | 1358.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1297.10 | 1265.49 | 1299.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 13:15:00 | 1225.15 | 1298.50 | 1311.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 09:15:00 | 1296.20 | 1258.25 | 1283.49 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 1376.85 | 1294.17 | 1294.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1406.10 | 1298.48 | 1296.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1317.05 | 1319.34 | 1307.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 14:15:00 | 1378.80 | 1318.03 | 1308.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1474.90 | 1435.05 | 1394.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-21 13:15:00 | 1504.20 | 1435.74 | 1394.96 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1697.50 | 1750.73 | 1683.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 09:15:00 | 1675.10 | 1748.53 | 1684.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1647.10 | 1697.67 | 1697.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1591.70 | 1692.80 | 1695.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1627.00 | 1620.33 | 1646.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 11:15:00 | 1596.80 | 1619.42 | 1642.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1594.20 | 1612.65 | 1635.79 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-10 10:15:00 | 1584.20 | 1612.13 | 1635.30 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1615.50 | 1593.48 | 1619.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-20 11:15:00 | 1630.50 | 1593.98 | 1619.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-14 13:15:00 | 1225.15 | 2025-03-06 09:15:00 | 1296.20 | EXIT_EMA400 | -71.05 |
| BUY | 2025-04-11 14:15:00 | 1378.80 | 2025-05-27 09:15:00 | 1589.07 | TARGET | 210.27 |
| BUY | 2025-05-21 13:15:00 | 1504.20 | 2025-06-12 10:15:00 | 1831.93 | TARGET | 327.73 |
| SELL | 2025-10-31 11:15:00 | 1596.80 | 2025-11-20 11:15:00 | 1630.50 | EXIT_EMA400 | -33.70 |
| SELL | 2025-11-10 10:15:00 | 1584.20 | 2025-11-20 11:15:00 | 1630.50 | EXIT_EMA400 | -46.30 |
