# Bharat Dynamics Ltd. (BDL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1373.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 202.41
- **Avg P&L per closed trade:** 33.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 10:15:00 | 1312.10 | 1391.46 | 1391.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 1302.60 | 1382.24 | 1386.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 10:15:00 | 1202.95 | 1194.11 | 1255.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-15 09:15:00 | 1172.55 | 1196.61 | 1251.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-27 14:15:00 | 1116.45 | 1050.35 | 1116.06 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 13:15:00 | 1280.00 | 1152.75 | 1152.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 1340.35 | 1173.74 | 1167.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 1184.10 | 1201.30 | 1183.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-28 12:15:00 | 1241.20 | 1200.41 | 1184.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-03 09:15:00 | 1170.95 | 1218.20 | 1196.00 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 1037.55 | 1183.80 | 1184.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 10:15:00 | 1029.80 | 1182.27 | 1183.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 1119.05 | 1092.87 | 1128.84 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 10:15:00 | 1307.05 | 1146.83 | 1146.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-25 13:15:00 | 1317.65 | 1151.74 | 1149.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1853.00 | 1853.14 | 1710.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 12:15:00 | 1866.90 | 1849.57 | 1721.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1795.00 | 1883.62 | 1793.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-17 09:15:00 | 1790.90 | 1882.70 | 1793.19 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 1590.40 | 1741.57 | 1741.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 1574.50 | 1739.91 | 1740.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 1570.40 | 1527.44 | 1593.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 12:15:00 | 1521.60 | 1560.51 | 1593.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 1539.00 | 1532.71 | 1563.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 09:15:00 | 1537.30 | 1533.16 | 1563.40 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1550.20 | 1533.66 | 1561.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 10:15:00 | 1539.40 | 1533.71 | 1561.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1539.90 | 1516.62 | 1543.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-12 09:15:00 | 1552.50 | 1517.92 | 1542.89 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 1356.50 | 1333.43 | 1333.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 1373.00 | 1334.10 | 1333.67 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-15 09:15:00 | 1172.55 | 2024-11-18 15:15:00 | 936.76 | TARGET | 235.79 |
| BUY | 2025-01-28 12:15:00 | 1241.20 | 2025-02-03 09:15:00 | 1170.95 | EXIT_EMA400 | -70.25 |
| BUY | 2025-06-27 12:15:00 | 1866.90 | 2025-07-17 09:15:00 | 1790.90 | EXIT_EMA400 | -76.00 |
| SELL | 2025-10-20 09:15:00 | 1537.30 | 2025-11-06 09:15:00 | 1459.01 | TARGET | 78.29 |
| SELL | 2025-10-24 10:15:00 | 1539.40 | 2025-11-06 09:15:00 | 1473.92 | TARGET | 65.48 |
| SELL | 2025-09-26 12:15:00 | 1521.60 | 2025-11-12 09:15:00 | 1552.50 | EXIT_EMA400 | -30.90 |
