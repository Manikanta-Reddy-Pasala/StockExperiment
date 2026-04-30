# PVR INOX Ltd. (PVRINOX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1074.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 13.73
- **Avg P&L per closed trade:** 4.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1478.15 | 1551.20 | 1551.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 1449.55 | 1540.57 | 1546.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 1512.95 | 1507.97 | 1525.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 14:15:00 | 1476.00 | 1527.97 | 1532.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-17 09:15:00 | 1523.35 | 1511.05 | 1522.77 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 1014.60 | 984.77 | 984.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1024.45 | 986.27 | 985.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 987.35 | 989.51 | 987.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 09:15:00 | 999.15 | 988.66 | 987.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 988.70 | 988.83 | 987.12 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-30 15:15:00 | 986.50 | 988.80 | 987.12 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1088.00 | 1106.55 | 1106.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 1080.50 | 1106.13 | 1106.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1119.40 | 1095.93 | 1100.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 09:15:00 | 1076.00 | 1095.58 | 1100.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 13:15:00 | 1025.95 | 994.46 | 1025.23 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-09 14:15:00 | 1476.00 | 2024-12-17 09:15:00 | 1523.35 | EXIT_EMA400 | -47.35 |
| BUY | 2025-07-30 09:15:00 | 999.15 | 2025-07-30 15:15:00 | 986.50 | EXIT_EMA400 | -12.65 |
| SELL | 2025-12-16 09:15:00 | 1076.00 | 2025-12-26 11:15:00 | 1002.27 | TARGET | 73.73 |
