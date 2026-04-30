# Aster DM Healthcare Ltd. (ASTERDM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 707.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 217.14
- **Avg P&L per closed trade:** 36.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 381.05 | 346.64 | 346.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 10:15:00 | 382.20 | 348.29 | 347.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 407.40 | 409.44 | 395.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 10:15:00 | 414.50 | 409.34 | 396.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 405.15 | 415.54 | 403.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-23 14:15:00 | 401.15 | 414.85 | 403.92 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 431.90 | 476.14 | 476.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 422.40 | 475.61 | 475.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 12:15:00 | 437.45 | 432.10 | 447.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-17 14:15:00 | 427.65 | 432.12 | 446.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 440.00 | 432.00 | 443.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 15:15:00 | 443.55 | 432.58 | 443.54 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 481.50 | 451.14 | 451.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 485.00 | 451.48 | 451.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 12:15:00 | 557.90 | 558.68 | 536.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 10:15:00 | 569.85 | 558.87 | 536.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 576.55 | 591.54 | 573.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 09:15:00 | 583.70 | 591.46 | 573.70 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-08 13:15:00 | 578.35 | 592.50 | 578.93 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 10:15:00 | 621.85 | 656.02 | 656.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 610.30 | 653.87 | 654.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 625.35 | 623.60 | 635.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 10:15:00 | 618.00 | 623.55 | 634.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 11:15:00 | 606.80 | 575.96 | 596.79 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 654.70 | 608.75 | 608.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 11:15:00 | 662.90 | 615.07 | 611.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 639.45 | 641.15 | 628.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-25 10:15:00 | 650.80 | 638.65 | 628.56 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-08 10:15:00 | 414.50 | 2024-10-23 14:15:00 | 401.15 | EXIT_EMA400 | -13.35 |
| SELL | 2025-03-17 14:15:00 | 427.65 | 2025-03-25 15:15:00 | 443.55 | EXIT_EMA400 | -15.90 |
| BUY | 2025-06-23 10:15:00 | 569.85 | 2025-07-03 15:15:00 | 669.05 | TARGET | 99.20 |
| BUY | 2025-07-29 09:15:00 | 583.70 | 2025-07-31 09:15:00 | 613.70 | TARGET | 30.00 |
| SELL | 2026-01-07 10:15:00 | 618.00 | 2026-01-19 15:15:00 | 567.54 | TARGET | 50.46 |
| BUY | 2026-03-25 10:15:00 | 650.80 | 2026-04-27 09:15:00 | 717.52 | TARGET | 66.72 |
