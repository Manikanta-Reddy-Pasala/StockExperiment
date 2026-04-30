# Bajaj Housing Finance Ltd. (BAJAJHFL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-09-16 09:15:00 → 2026-04-30 15:15:00 (2804 bars)
- **Last close:** 87.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| EXIT | 2 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -9.03
- **Avg P&L per closed trade:** -2.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 131.02 | 119.13 | 119.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 131.68 | 120.98 | 120.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 123.27 | 123.36 | 121.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-19 09:15:00 | 125.14 | 122.24 | 121.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-30 11:15:00 | 122.17 | 123.09 | 122.20 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 121.52 | 122.13 | 122.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 121.32 | 122.13 | 122.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 122.99 | 121.99 | 122.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-07 09:15:00 | 121.03 | 122.00 | 122.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 121.03 | 122.00 | 122.06 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-08 10:15:00 | 120.43 | 121.92 | 122.02 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 121.52 | 121.85 | 121.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-14 09:15:00 | 120.10 | 121.71 | 121.90 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 120.90 | 121.60 | 121.84 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-16 10:15:00 | 122.54 | 121.59 | 121.82 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-19 09:15:00 | 125.14 | 2025-05-30 11:15:00 | 122.17 | EXIT_EMA400 | -2.97 |
| SELL | 2025-07-07 09:15:00 | 121.03 | 2025-07-16 10:15:00 | 122.54 | EXIT_EMA400 | -1.51 |
| SELL | 2025-07-08 10:15:00 | 120.43 | 2025-07-16 10:15:00 | 122.54 | EXIT_EMA400 | -2.11 |
| SELL | 2025-07-14 09:15:00 | 120.10 | 2025-07-16 10:15:00 | 122.54 | EXIT_EMA400 | -2.44 |
