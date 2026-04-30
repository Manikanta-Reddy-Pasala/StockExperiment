# Adani Total Gas Ltd. (ATGL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 634.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -101.05
- **Avg P&L per closed trade:** -25.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 658.25 | 620.53 | 620.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 673.10 | 621.84 | 621.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 662.45 | 667.52 | 651.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 685.65 | 654.00 | 648.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 685.65 | 654.00 | 648.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-27 10:15:00 | 689.90 | 654.35 | 648.40 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-08 10:15:00 | 652.15 | 658.88 | 652.35 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 629.00 | 649.76 | 649.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 625.00 | 648.54 | 649.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 625.20 | 623.03 | 633.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 14:15:00 | 618.00 | 623.95 | 632.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 631.30 | 623.65 | 632.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-26 09:15:00 | 636.40 | 624.00 | 632.25 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 742.50 | 628.59 | 628.13 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 621.00 | 631.05 | 631.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 618.55 | 628.83 | 629.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 658.10 | 628.56 | 629.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 09:15:00 | 618.15 | 629.71 | 630.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-12 09:15:00 | 629.55 | 625.69 | 627.98 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 600.10 | 542.94 | 542.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 617.10 | 543.68 | 543.16 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-27 09:15:00 | 685.65 | 2025-07-08 10:15:00 | 652.15 | EXIT_EMA400 | -33.50 |
| BUY | 2025-06-27 10:15:00 | 689.90 | 2025-07-08 10:15:00 | 652.15 | EXIT_EMA400 | -37.75 |
| SELL | 2025-08-21 14:15:00 | 618.00 | 2025-08-26 09:15:00 | 636.40 | EXIT_EMA400 | -18.40 |
| SELL | 2025-11-06 09:15:00 | 618.15 | 2025-11-12 09:15:00 | 629.55 | EXIT_EMA400 | -11.40 |
