# PG Electroplast Ltd. (PGEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 535.50
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
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 123.68
- **Avg P&L per closed trade:** 30.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 793.90 | 859.41 | 859.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 772.00 | 848.96 | 854.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 768.25 | 763.95 | 787.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-07 09:15:00 | 736.10 | 787.20 | 791.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 590.15 | 561.38 | 595.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 12:15:00 | 586.50 | 561.63 | 595.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 566.25 | 557.81 | 579.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-14 15:15:00 | 580.00 | 558.42 | 579.34 | Close above EMA400 |

### Cycle 2 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 628.30 | 578.84 | 578.63 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 10:15:00 | 533.65 | 580.06 | 580.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 11:15:00 | 528.85 | 579.55 | 580.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 562.60 | 561.82 | 569.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 13:15:00 | 547.50 | 561.89 | 569.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 584.00 | 562.13 | 569.53 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 624.25 | 575.15 | 575.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 628.65 | 584.32 | 579.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 597.30 | 599.95 | 590.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 14:15:00 | 612.55 | 599.15 | 590.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 579.70 | 599.93 | 591.11 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 513.05 | 583.49 | 583.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 505.40 | 582.71 | 583.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 10:15:00 | 517.35 | 511.44 | 537.24 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-08-07 09:15:00 | 736.10 | 2025-08-08 15:15:00 | 570.43 | TARGET | 165.67 |
| SELL | 2025-10-20 12:15:00 | 586.50 | 2025-11-06 10:15:00 | 559.15 | TARGET | 27.35 |
| SELL | 2026-02-03 13:15:00 | 547.50 | 2026-02-04 09:15:00 | 584.00 | EXIT_EMA400 | -36.50 |
| BUY | 2026-03-05 14:15:00 | 612.55 | 2026-03-09 09:15:00 | 579.70 | EXIT_EMA400 | -32.85 |
