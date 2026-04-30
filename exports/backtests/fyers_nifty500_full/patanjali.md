# Patanjali Foods Ltd. (PATANJALI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 458.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| EXIT | 1 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 0
- **Winners / losers:** 0 / 1
- **Target hits / EMA400 exits:** 0 / 1
- **Total realized P&L (per unit):** -14.20
- **Avg P&L per closed trade:** -14.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 10:15:00 | 571.00 | 601.46 | 601.55 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 604.48 | 596.59 | 596.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 619.38 | 597.05 | 596.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 13:15:00 | 623.47 | 626.45 | 614.97 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 572.37 | 608.86 | 608.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 554.83 | 594.44 | 600.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 556.17 | 556.13 | 569.14 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 650.53 | 578.28 | 578.05 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 584.00 | 595.72 | 595.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 579.00 | 594.57 | 595.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 595.85 | 593.00 | 594.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 10:15:00 | 584.25 | 593.00 | 594.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 591.95 | 592.18 | 593.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 09:15:00 | 598.45 | 592.18 | 593.61 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-10-24 10:15:00 | 584.25 | 2025-10-29 09:15:00 | 598.45 | EXIT_EMA400 | -14.20 |
