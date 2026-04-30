# TBO Tek Ltd. (TBOTEK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-15 09:15:00 → 2026-04-30 15:15:00 (3395 bars)
- **Last close:** 1245.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -92.04
- **Avg P&L per closed trade:** -15.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 1600.00 | 1717.93 | 1718.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 13:15:00 | 1590.15 | 1716.66 | 1717.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 1679.85 | 1663.37 | 1686.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 14:15:00 | 1592.00 | 1668.62 | 1686.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-11 11:15:00 | 1627.75 | 1578.77 | 1620.13 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 1748.65 | 1645.14 | 1644.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 1758.05 | 1653.71 | 1649.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 09:15:00 | 1639.85 | 1700.16 | 1677.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 1725.20 | 1697.50 | 1676.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1725.20 | 1697.50 | 1676.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-17 13:15:00 | 1674.25 | 1696.83 | 1678.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 14:15:00 | 1623.20 | 1664.81 | 1664.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 1440.20 | 1580.81 | 1612.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 1172.40 | 1131.06 | 1235.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-09 09:15:00 | 1115.40 | 1132.68 | 1232.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1216.70 | 1149.68 | 1225.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-16 10:15:00 | 1227.10 | 1150.45 | 1225.91 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 1363.90 | 1258.51 | 1258.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 14:15:00 | 1386.30 | 1277.55 | 1269.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 1318.40 | 1331.59 | 1303.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-10 13:15:00 | 1342.00 | 1331.58 | 1303.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1340.90 | 1371.28 | 1339.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-31 10:15:00 | 1357.90 | 1371.14 | 1339.65 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-05 09:15:00 | 1339.60 | 1374.67 | 1344.48 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1478.30 | 1607.64 | 1607.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1465.20 | 1606.22 | 1607.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1495.00 | 1494.71 | 1537.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 10:15:00 | 1475.00 | 1504.07 | 1537.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-13 14:15:00 | 1538.30 | 1500.38 | 1534.07 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-13 14:15:00 | 1592.00 | 2024-12-11 11:15:00 | 1627.75 | EXIT_EMA400 | -35.75 |
| BUY | 2025-01-14 10:15:00 | 1725.20 | 2025-01-17 13:15:00 | 1674.25 | EXIT_EMA400 | -50.95 |
| SELL | 2025-05-09 09:15:00 | 1115.40 | 2025-05-16 10:15:00 | 1227.10 | EXIT_EMA400 | -111.70 |
| BUY | 2025-07-10 13:15:00 | 1342.00 | 2025-07-23 13:15:00 | 1456.91 | TARGET | 114.91 |
| BUY | 2025-07-31 10:15:00 | 1357.90 | 2025-08-01 10:15:00 | 1412.65 | TARGET | 54.75 |
| SELL | 2026-02-12 10:15:00 | 1475.00 | 2026-02-13 14:15:00 | 1538.30 | EXIT_EMA400 | -63.30 |
