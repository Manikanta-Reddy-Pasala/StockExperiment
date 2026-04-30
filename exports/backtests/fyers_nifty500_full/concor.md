# Container Corporation of India Ltd. (CONCOR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 508.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -26.36
- **Avg P&L per closed trade:** -13.18

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 585.24 | 561.70 | 561.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 589.92 | 561.98 | 561.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 604.40 | 604.91 | 588.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 613.12 | 605.15 | 588.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 589.36 | 604.56 | 590.01 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 537.50 | 595.03 | 595.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 534.00 | 594.42 | 594.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 561.90 | 553.67 | 565.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 09:15:00 | 545.40 | 555.36 | 564.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-29 11:15:00 | 548.00 | 538.44 | 547.17 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 518.40 | 486.83 | 486.75 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-16 13:15:00 | 613.12 | 2025-06-19 10:15:00 | 589.36 | EXIT_EMA400 | -23.76 |
| SELL | 2025-09-23 09:15:00 | 545.40 | 2025-10-29 11:15:00 | 548.00 | EXIT_EMA400 | -2.60 |
