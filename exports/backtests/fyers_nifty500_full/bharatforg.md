# Bharat Forge Ltd. (BHARATFORG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1890.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 11.16
- **Avg P&L per closed trade:** 3.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 11:15:00 | 1568.80 | 1594.01 | 1594.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1565.80 | 1593.50 | 1593.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 1592.65 | 1592.36 | 1593.24 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1599.55 | 1594.09 | 1594.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 10:15:00 | 1605.75 | 1594.32 | 1594.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 1574.95 | 1594.15 | 1594.11 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 1568.30 | 1593.89 | 1593.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1558.65 | 1593.14 | 1593.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1597.50 | 1591.63 | 1592.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-23 13:15:00 | 1577.75 | 1591.55 | 1592.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 09:15:00 | 1161.15 | 1093.82 | 1149.23 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1248.00 | 1131.84 | 1131.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1263.50 | 1135.38 | 1133.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 12:15:00 | 1272.40 | 1274.93 | 1237.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-01 14:15:00 | 1283.60 | 1275.04 | 1237.67 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 10:15:00 | 1242.60 | 1281.37 | 1248.91 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1167.90 | 1232.53 | 1232.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1161.90 | 1231.82 | 1232.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1199.90 | 1160.49 | 1183.68 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 1270.90 | 1198.34 | 1198.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1302.40 | 1221.22 | 1212.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1374.90 | 1375.93 | 1327.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 11:15:00 | 1388.80 | 1375.81 | 1329.29 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-20 09:15:00 | 1396.10 | 1439.32 | 1402.37 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-23 13:15:00 | 1577.75 | 2024-09-26 10:15:00 | 1532.89 | TARGET | 44.86 |
| BUY | 2025-07-01 14:15:00 | 1283.60 | 2025-07-10 10:15:00 | 1242.60 | EXIT_EMA400 | -41.00 |
| BUY | 2025-12-09 11:15:00 | 1388.80 | 2026-01-20 09:15:00 | 1396.10 | EXIT_EMA400 | 7.30 |
