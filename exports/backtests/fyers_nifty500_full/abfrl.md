# Aditya Birla Fashion and Retail Ltd. (ABFRL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 63.90
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
- **Winners / losers:** 2 / 0
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 41.36
- **Avg P&L per closed trade:** 20.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 306.35 | 324.29 | 324.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 304.40 | 323.92 | 324.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 306.00 | 305.36 | 312.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 09:15:00 | 298.75 | 308.45 | 311.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-01 13:15:00 | 287.85 | 277.43 | 286.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 278.25 | 262.39 | 262.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 281.15 | 263.98 | 263.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 90.45 | 262.21 | 262.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 90.00 | 260.49 | 261.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 79.94 | 77.73 | 95.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-24 14:15:00 | 75.38 | 80.63 | 83.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-09 11:15:00 | 73.63 | 69.66 | 73.11 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-13 09:15:00 | 298.75 | 2025-01-13 14:15:00 | 259.14 | TARGET | 39.61 |
| SELL | 2025-11-24 14:15:00 | 75.38 | 2026-02-09 11:15:00 | 73.63 | EXIT_EMA400 | 1.75 |
