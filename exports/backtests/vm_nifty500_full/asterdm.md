# Aster DM Healthcare Ltd. (ASTERDM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 701.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 210.57
- **Avg P&L per closed trade:** 30.08

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 13:15:00 | 350.00 | 435.42 | 435.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 349.20 | 434.56 | 435.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 379.55 | 377.99 | 397.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-29 09:15:00 | 372.70 | 378.05 | 397.11 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-21 15:15:00 | 380.00 | 363.06 | 378.82 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 11:15:00 | 388.00 | 363.25 | 363.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 388.35 | 364.46 | 363.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 407.40 | 409.48 | 397.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 10:15:00 | 414.75 | 409.39 | 398.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-22 14:15:00 | 405.15 | 415.55 | 405.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 418.00 | 475.72 | 475.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 414.85 | 473.97 | 475.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 12:15:00 | 437.85 | 432.30 | 447.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-17 14:15:00 | 427.65 | 432.29 | 446.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 440.00 | 432.15 | 443.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 15:15:00 | 444.70 | 432.73 | 443.67 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 482.00 | 451.24 | 451.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 485.00 | 451.58 | 451.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 12:15:00 | 558.40 | 558.72 | 536.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 10:15:00 | 569.85 | 558.91 | 536.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 576.55 | 591.56 | 573.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 09:15:00 | 583.70 | 591.49 | 573.73 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-08 13:15:00 | 578.35 | 592.51 | 578.95 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 10:15:00 | 621.85 | 656.05 | 656.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 610.30 | 653.90 | 654.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 625.35 | 623.57 | 635.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 10:15:00 | 617.60 | 623.53 | 634.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 11:15:00 | 606.80 | 576.86 | 597.95 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 643.60 | 609.91 | 609.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 15:15:00 | 644.60 | 611.75 | 610.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 639.45 | 641.23 | 628.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-25 10:15:00 | 650.20 | 638.65 | 628.90 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-29 09:15:00 | 372.70 | 2024-06-21 15:15:00 | 380.00 | EXIT_EMA400 | -7.30 |
| BUY | 2024-10-08 10:15:00 | 414.75 | 2024-10-22 14:15:00 | 405.15 | EXIT_EMA400 | -9.60 |
| SELL | 2025-03-17 14:15:00 | 427.65 | 2025-03-25 15:15:00 | 444.70 | EXIT_EMA400 | -17.05 |
| BUY | 2025-06-23 10:15:00 | 569.85 | 2025-07-03 15:15:00 | 668.93 | TARGET | 99.08 |
| BUY | 2025-07-29 09:15:00 | 583.70 | 2025-07-31 09:15:00 | 613.60 | TARGET | 29.90 |
| SELL | 2026-01-07 10:15:00 | 617.60 | 2026-01-20 09:15:00 | 565.95 | TARGET | 51.65 |
| BUY | 2026-03-25 10:15:00 | 650.20 | 2026-04-27 09:15:00 | 714.09 | TARGET | 63.89 |
