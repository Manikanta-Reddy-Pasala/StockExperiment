# ICICI Prudential Life Insurance Company Ltd. (ICICIPRULI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 514.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 99.88
- **Avg P&L per closed trade:** 24.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 690.20 | 729.81 | 729.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 09:15:00 | 685.95 | 727.62 | 728.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 676.15 | 674.04 | 689.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 660.80 | 673.89 | 689.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 596.00 | 569.11 | 593.51 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 613.30 | 591.81 | 591.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 617.20 | 593.07 | 592.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 627.00 | 629.27 | 616.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 629.65 | 629.11 | 616.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 636.80 | 650.31 | 637.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 615.00 | 630.43 | 630.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 13:15:00 | 612.35 | 629.36 | 629.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 10:15:00 | 628.90 | 625.95 | 628.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-13 13:15:00 | 621.45 | 625.87 | 627.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-14 09:15:00 | 629.90 | 625.82 | 627.91 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 630.35 | 608.33 | 608.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 631.00 | 614.74 | 612.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 15:15:00 | 660.50 | 660.98 | 645.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-04 12:15:00 | 661.90 | 651.58 | 644.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-06 13:15:00 | 645.50 | 651.99 | 645.50 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 600.60 | 644.37 | 644.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 596.40 | 638.86 | 641.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 565.10 | 564.36 | 591.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-15 11:15:00 | 562.45 | 564.36 | 591.49 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-06 10:15:00 | 660.80 | 2025-01-22 09:15:00 | 574.47 | TARGET | 86.33 |
| BUY | 2025-06-13 14:15:00 | 629.65 | 2025-07-08 09:15:00 | 668.05 | TARGET | 38.40 |
| SELL | 2025-08-13 13:15:00 | 621.45 | 2025-08-14 09:15:00 | 629.90 | EXIT_EMA400 | -8.45 |
| BUY | 2026-02-04 12:15:00 | 661.90 | 2026-02-06 13:15:00 | 645.50 | EXIT_EMA400 | -16.40 |
