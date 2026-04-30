# Afcons Infrastructure Ltd. (AFCONS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-04 09:15:00 → 2026-04-30 15:15:00 (2553 bars)
- **Last close:** 339.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -28.30
- **Avg P&L per closed trade:** -14.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 451.10 | 499.36 | 499.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 444.25 | 495.83 | 497.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 14:15:00 | 450.45 | 448.28 | 464.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-17 12:15:00 | 442.55 | 450.59 | 463.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 457.90 | 450.57 | 462.54 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 09:15:00 | 463.05 | 450.99 | 462.34 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 438.95 | 427.25 | 427.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 446.25 | 427.66 | 427.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 440.20 | 441.92 | 436.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 451.25 | 441.86 | 436.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 443.85 | 450.17 | 443.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-20 11:15:00 | 443.45 | 450.04 | 443.49 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 408.00 | 441.43 | 441.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 12:15:00 | 404.35 | 438.19 | 439.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 10:15:00 | 291.60 | 290.74 | 312.95 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-17 12:15:00 | 442.55 | 2025-03-20 09:15:00 | 463.05 | EXIT_EMA400 | -20.50 |
| BUY | 2025-09-29 09:15:00 | 451.25 | 2025-10-20 11:15:00 | 443.45 | EXIT_EMA400 | -7.80 |
