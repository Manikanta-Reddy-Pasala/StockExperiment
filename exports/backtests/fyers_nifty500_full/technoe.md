# Techno Electric & Engineering Company Ltd. (TECHNOE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1282.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -433.15
- **Avg P&L per closed trade:** -86.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 10:15:00 | 1448.30 | 1586.99 | 1587.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 13:15:00 | 1439.70 | 1583.31 | 1585.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 10:15:00 | 1548.05 | 1522.84 | 1548.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 11:15:00 | 1490.90 | 1522.98 | 1546.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-17 09:15:00 | 1544.30 | 1498.90 | 1528.14 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1682.65 | 1546.70 | 1546.21 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 1395.90 | 1546.97 | 1547.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1387.80 | 1543.96 | 1546.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 994.10 | 990.44 | 1108.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 909.25 | 1003.75 | 1076.26 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1034.40 | 994.45 | 1056.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 11:15:00 | 1078.90 | 995.90 | 1057.01 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 1256.60 | 1083.58 | 1082.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 1274.30 | 1089.03 | 1085.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1525.10 | 1525.92 | 1435.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-22 09:15:00 | 1547.00 | 1462.16 | 1439.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1472.00 | 1498.82 | 1470.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-10 12:15:00 | 1468.00 | 1498.51 | 1470.06 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1309.90 | 1454.30 | 1454.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1298.10 | 1452.74 | 1453.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 1039.40 | 1008.26 | 1086.24 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 09:15:00 | 1117.80 | 1100.19 | 1100.18 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1033.70 | 1100.18 | 1100.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1028.00 | 1099.46 | 1099.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 1092.90 | 1092.08 | 1095.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 1035.90 | 1091.16 | 1095.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1075.60 | 1072.01 | 1083.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-09 09:15:00 | 1064.00 | 1072.33 | 1083.62 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 1115.50 | 1072.85 | 1083.50 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 1205.00 | 1092.67 | 1092.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1219.00 | 1093.93 | 1093.19 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-06 11:15:00 | 1490.90 | 2024-12-17 09:15:00 | 1544.30 | EXIT_EMA400 | -53.40 |
| SELL | 2025-04-07 09:15:00 | 909.25 | 2025-04-17 11:15:00 | 1078.90 | EXIT_EMA400 | -169.65 |
| BUY | 2025-08-22 09:15:00 | 1547.00 | 2025-09-10 12:15:00 | 1468.00 | EXIT_EMA400 | -79.00 |
| SELL | 2026-03-27 09:15:00 | 1035.90 | 2026-04-10 09:15:00 | 1115.50 | EXIT_EMA400 | -79.60 |
| SELL | 2026-04-09 09:15:00 | 1064.00 | 2026-04-10 09:15:00 | 1115.50 | EXIT_EMA400 | -51.50 |
