# India Cements Ltd. (INDIACEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 393.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -37.95
- **Avg P&L per closed trade:** -12.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 333.25 | 354.64 | 354.69 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 373.55 | 354.31 | 354.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 376.45 | 354.53 | 354.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 368.25 | 370.43 | 364.69 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 297.70 | 359.71 | 359.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 276.10 | 357.00 | 358.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 12:15:00 | 280.90 | 279.98 | 303.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 09:15:00 | 276.35 | 283.08 | 300.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 287.95 | 278.27 | 288.11 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-16 13:15:00 | 283.20 | 278.31 | 288.09 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-17 09:15:00 | 294.40 | 278.57 | 288.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 13:15:00 | 315.10 | 292.78 | 292.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 319.00 | 294.97 | 293.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 312.65 | 315.32 | 306.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 09:15:00 | 325.00 | 315.41 | 306.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 319.00 | 327.17 | 317.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-20 09:15:00 | 316.30 | 326.83 | 317.51 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 387.10 | 432.68 | 432.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 382.55 | 426.25 | 429.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 376.75 | 375.97 | 394.82 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-13 09:15:00 | 276.35 | 2025-04-17 09:15:00 | 294.40 | EXIT_EMA400 | -18.05 |
| SELL | 2025-04-16 13:15:00 | 283.20 | 2025-04-17 09:15:00 | 294.40 | EXIT_EMA400 | -11.20 |
| BUY | 2025-06-02 09:15:00 | 325.00 | 2025-06-20 09:15:00 | 316.30 | EXIT_EMA400 | -8.70 |
