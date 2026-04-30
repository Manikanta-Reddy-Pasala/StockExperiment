# Tata Teleservices (Maharashtra) Ltd. (TTML.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 42.89
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -12.73
- **Avg P&L per closed trade:** -6.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 81.80 | 88.70 | 88.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 78.76 | 88.21 | 88.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 74.96 | 72.91 | 77.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 14:15:00 | 69.45 | 76.76 | 77.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-20 09:15:00 | 80.76 | 74.75 | 76.55 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 75.37 | 62.42 | 62.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 80.12 | 67.97 | 65.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 67.81 | 69.37 | 66.67 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 63.59 | 65.93 | 65.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 62.52 | 65.87 | 65.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 59.99 | 58.36 | 60.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 14:15:00 | 57.68 | 58.35 | 59.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-07 12:15:00 | 59.10 | 57.22 | 58.78 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-10 14:15:00 | 69.45 | 2025-01-20 09:15:00 | 80.76 | EXIT_EMA400 | -11.31 |
| SELL | 2025-09-19 14:15:00 | 57.68 | 2025-10-07 12:15:00 | 59.10 | EXIT_EMA400 | -1.42 |
