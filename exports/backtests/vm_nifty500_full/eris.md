# Eris Lifesciences Ltd. (ERIS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1322.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 525.63
- **Avg P&L per closed trade:** 65.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 855.60 | 894.90 | 894.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 845.70 | 887.74 | 890.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 858.00 | 856.23 | 867.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-09 13:15:00 | 846.15 | 871.12 | 872.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 870.35 | 867.95 | 870.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-14 13:15:00 | 875.65 | 868.05 | 870.61 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 15:15:00 | 910.00 | 872.79 | 872.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 11:15:00 | 915.70 | 883.34 | 878.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 1010.30 | 1010.42 | 973.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-16 13:15:00 | 1017.55 | 1010.46 | 974.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1253.55 | 1328.51 | 1249.89 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-27 12:15:00 | 1263.55 | 1327.14 | 1249.98 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1303.90 | 1348.72 | 1296.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-22 11:15:00 | 1290.00 | 1348.13 | 1295.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 1284.60 | 1360.44 | 1360.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1253.85 | 1356.46 | 1358.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1297.10 | 1268.00 | 1302.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 13:15:00 | 1225.15 | 1300.00 | 1313.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 09:15:00 | 1296.20 | 1258.49 | 1284.49 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 1372.90 | 1295.78 | 1295.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 1389.75 | 1296.72 | 1295.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1317.05 | 1319.63 | 1308.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 14:15:00 | 1378.80 | 1318.47 | 1309.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1474.90 | 1435.16 | 1394.69 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-21 13:15:00 | 1504.20 | 1435.84 | 1395.24 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1697.50 | 1750.72 | 1683.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 09:15:00 | 1675.10 | 1748.38 | 1684.69 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1647.10 | 1697.76 | 1697.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1591.70 | 1692.89 | 1695.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1627.00 | 1620.41 | 1646.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 11:15:00 | 1596.80 | 1619.57 | 1642.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1594.20 | 1612.67 | 1635.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-10 10:15:00 | 1584.20 | 1612.15 | 1635.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1615.50 | 1593.74 | 1619.28 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-20 11:15:00 | 1630.50 | 1594.23 | 1619.28 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-09 13:15:00 | 846.15 | 2024-05-14 13:15:00 | 875.65 | EXIT_EMA400 | -29.50 |
| BUY | 2024-07-16 13:15:00 | 1017.55 | 2024-08-07 11:15:00 | 1147.88 | TARGET | 130.33 |
| BUY | 2024-09-27 12:15:00 | 1263.55 | 2024-09-30 09:15:00 | 1304.26 | TARGET | 40.71 |
| SELL | 2025-02-14 13:15:00 | 1225.15 | 2025-03-06 09:15:00 | 1296.20 | EXIT_EMA400 | -71.05 |
| BUY | 2025-04-11 14:15:00 | 1378.80 | 2025-05-27 09:15:00 | 1587.06 | TARGET | 208.26 |
| BUY | 2025-05-21 13:15:00 | 1504.20 | 2025-06-12 10:15:00 | 1831.08 | TARGET | 326.88 |
| SELL | 2025-10-31 11:15:00 | 1596.80 | 2025-11-20 11:15:00 | 1630.50 | EXIT_EMA400 | -33.70 |
| SELL | 2025-11-10 10:15:00 | 1584.20 | 2025-11-20 11:15:00 | 1630.50 | EXIT_EMA400 | -46.30 |
