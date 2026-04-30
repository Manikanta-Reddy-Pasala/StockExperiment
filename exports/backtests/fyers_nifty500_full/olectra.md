# Olectra Greentech Ltd. (OLECTRA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1239.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 250.80
- **Avg P&L per closed trade:** 62.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 13:15:00 | 1586.60 | 1724.09 | 1724.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 11:15:00 | 1573.80 | 1717.25 | 1721.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 1723.90 | 1690.82 | 1707.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-16 14:15:00 | 1611.85 | 1682.77 | 1701.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1617.35 | 1602.00 | 1641.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-12 09:15:00 | 1641.70 | 1602.54 | 1641.35 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 15:15:00 | 1745.00 | 1656.86 | 1656.67 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 15:15:00 | 1619.50 | 1660.38 | 1660.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 1614.75 | 1657.47 | 1658.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 1559.95 | 1550.98 | 1594.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 1537.05 | 1574.33 | 1591.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-30 10:15:00 | 1477.20 | 1412.93 | 1468.88 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1336.40 | 1222.10 | 1221.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 15:15:00 | 1350.00 | 1225.65 | 1223.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1213.00 | 1228.87 | 1225.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 09:15:00 | 1252.90 | 1227.43 | 1224.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1228.60 | 1229.02 | 1225.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-05 14:15:00 | 1221.00 | 1229.31 | 1226.08 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1172.30 | 1223.82 | 1223.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1158.00 | 1213.80 | 1218.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.12 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 1278.30 | 1213.39 | 1213.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 1284.20 | 1217.35 | 1215.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 1573.20 | 1578.42 | 1495.74 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 1412.40 | 1486.59 | 1486.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1391.10 | 1481.20 | 1484.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 1261.00 | 1257.13 | 1329.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 10:15:00 | 1209.00 | 1254.52 | 1323.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-18 14:15:00 | 1060.25 | 974.94 | 1048.07 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 12:15:00 | 1227.05 | 1071.79 | 1071.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 1232.60 | 1095.51 | 1084.09 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-16 14:15:00 | 1611.85 | 2024-09-12 09:15:00 | 1641.70 | EXIT_EMA400 | -29.85 |
| SELL | 2024-12-18 09:15:00 | 1537.05 | 2025-01-10 09:15:00 | 1373.25 | TARGET | 163.80 |
| BUY | 2025-06-02 09:15:00 | 1252.90 | 2025-06-05 14:15:00 | 1221.00 | EXIT_EMA400 | -31.90 |
| SELL | 2025-12-29 10:15:00 | 1209.00 | 2026-03-18 14:15:00 | 1060.25 | EXIT_EMA400 | 148.75 |
