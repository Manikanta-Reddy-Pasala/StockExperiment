# Amara Raja Energy & Mobility Ltd. (ARE&M.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 878.95
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
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 2 / 1
- **Total realized P&L (per unit):** 173.60
- **Avg P&L per closed trade:** 57.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 1374.05 | 1474.36 | 1474.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 1363.50 | 1473.26 | 1473.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 10:15:00 | 1408.20 | 1406.52 | 1431.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-14 09:15:00 | 1393.00 | 1406.41 | 1431.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-31 14:15:00 | 1397.05 | 1357.50 | 1392.17 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 1019.00 | 985.81 | 985.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1027.85 | 986.87 | 986.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 1010.00 | 1010.54 | 1000.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-23 11:15:00 | 1013.15 | 1010.34 | 1000.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-24 10:15:00 | 996.00 | 1010.29 | 1000.98 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 965.50 | 997.67 | 997.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 961.00 | 997.30 | 997.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 937.60 | 933.12 | 950.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-05 14:15:00 | 924.30 | 933.20 | 949.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 897.50 | 873.02 | 899.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-10 09:15:00 | 907.00 | 873.59 | 899.92 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-14 09:15:00 | 1393.00 | 2024-10-22 10:15:00 | 1278.94 | TARGET | 114.06 |
| BUY | 2025-09-23 11:15:00 | 1013.15 | 2025-09-24 10:15:00 | 996.00 | EXIT_EMA400 | -17.15 |
| SELL | 2026-01-05 14:15:00 | 924.30 | 2026-01-21 09:15:00 | 847.61 | TARGET | 76.69 |
