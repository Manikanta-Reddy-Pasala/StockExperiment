# Swan Corp Ltd. (SWANCORP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-09-04 09:15:00 → 2026-04-30 15:15:00 (1111 bars)
- **Last close:** 334.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 1 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 2
- **Winners / losers:** 0 / 1
- **Target hits / EMA400 exits:** 0 / 1
- **Total realized P&L (per unit):** -15.45
- **Avg P&L per closed trade:** -15.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 470.70 | 455.96 | 455.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 476.00 | 456.81 | 456.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 461.50 | 462.30 | 459.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-05 09:15:00 | 474.70 | 462.42 | 459.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-08 10:15:00 | 459.25 | 463.80 | 460.71 | Close below EMA400 |

### Cycle 2 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 420.10 | 457.66 | 457.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 419.25 | 457.28 | 457.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 364.00 | 338.79 | 364.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-24 10:15:00 | 329.10 | 341.88 | 362.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 356.65 | 341.04 | 359.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-30 09:15:00 | 337.05 | 341.34 | 359.21 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2026-01-05 09:15:00 | 474.70 | 2026-01-08 10:15:00 | 459.25 | EXIT_EMA400 | -15.45 |
