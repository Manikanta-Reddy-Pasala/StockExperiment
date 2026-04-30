# Bharat Forge Ltd. (BHARATFORG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1881.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -96.59
- **Avg P&L per closed trade:** -13.80

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 11:15:00 | 1140.55 | 1179.02 | 1179.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 1134.30 | 1174.77 | 1176.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 10:15:00 | 1148.55 | 1144.31 | 1157.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-02 09:15:00 | 1140.30 | 1144.42 | 1157.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1140.30 | 1144.42 | 1157.59 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-03 09:15:00 | 1138.45 | 1144.89 | 1157.37 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-04 10:15:00 | 1159.80 | 1144.75 | 1156.81 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 13:15:00 | 1209.65 | 1163.47 | 1163.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 1216.00 | 1164.83 | 1164.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 1622.35 | 1627.20 | 1527.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-10 11:15:00 | 1655.80 | 1627.49 | 1528.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1590.75 | 1626.21 | 1552.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-26 09:15:00 | 1660.70 | 1621.22 | 1556.36 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 1576.15 | 1652.09 | 1585.87 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1524.20 | 1580.68 | 1580.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1514.55 | 1577.29 | 1579.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 1470.00 | 1462.05 | 1501.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 14:15:00 | 1451.85 | 1462.37 | 1499.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 09:15:00 | 1161.15 | 1094.21 | 1150.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1248.00 | 1131.93 | 1131.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1263.50 | 1135.47 | 1133.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 1275.00 | 1275.02 | 1237.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-01 14:15:00 | 1283.60 | 1275.09 | 1237.82 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 10:15:00 | 1242.60 | 1281.40 | 1249.03 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1167.90 | 1232.55 | 1232.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1161.20 | 1231.84 | 1232.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1199.90 | 1160.51 | 1183.72 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 1270.90 | 1198.38 | 1198.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1302.40 | 1221.33 | 1212.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1374.60 | 1375.85 | 1327.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 11:15:00 | 1388.80 | 1375.77 | 1329.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-20 09:15:00 | 1396.10 | 1439.36 | 1402.38 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-02 09:15:00 | 1140.30 | 2024-04-04 10:15:00 | 1159.80 | EXIT_EMA400 | -19.50 |
| SELL | 2024-04-03 09:15:00 | 1138.45 | 2024-04-04 10:15:00 | 1159.80 | EXIT_EMA400 | -21.35 |
| BUY | 2024-07-10 11:15:00 | 1655.80 | 2024-08-05 09:15:00 | 1576.15 | EXIT_EMA400 | -79.65 |
| BUY | 2024-07-26 09:15:00 | 1660.70 | 2024-08-05 09:15:00 | 1576.15 | EXIT_EMA400 | -84.55 |
| SELL | 2024-11-07 14:15:00 | 1451.85 | 2024-11-14 11:15:00 | 1309.69 | TARGET | 142.16 |
| BUY | 2025-07-01 14:15:00 | 1283.60 | 2025-07-10 10:15:00 | 1242.60 | EXIT_EMA400 | -41.00 |
| BUY | 2025-12-09 11:15:00 | 1388.80 | 2026-01-20 09:15:00 | 1396.10 | EXIT_EMA400 | 7.30 |
