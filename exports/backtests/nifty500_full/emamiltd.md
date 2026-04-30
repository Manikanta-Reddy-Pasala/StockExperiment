# Emami Ltd. (EMAMILTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 444.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -134.30
- **Avg P&L per closed trade:** -16.79

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 496.00 | 504.98 | 505.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 09:15:00 | 494.75 | 503.80 | 504.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 09:15:00 | 508.20 | 502.71 | 503.75 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 12:15:00 | 557.25 | 504.92 | 504.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 14:15:00 | 563.75 | 506.02 | 505.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 12:15:00 | 529.00 | 529.26 | 519.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-15 14:15:00 | 534.80 | 528.93 | 519.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-17 11:15:00 | 515.00 | 528.91 | 520.12 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 504.00 | 514.13 | 514.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 09:15:00 | 501.00 | 513.50 | 513.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 451.25 | 448.12 | 466.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-19 09:15:00 | 433.60 | 448.23 | 460.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 458.30 | 447.26 | 458.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-25 11:15:00 | 458.65 | 447.37 | 458.31 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 15:15:00 | 526.10 | 466.49 | 466.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 09:15:00 | 533.25 | 467.15 | 466.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 744.40 | 763.66 | 704.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 11:15:00 | 780.95 | 763.04 | 706.54 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-16 09:15:00 | 746.50 | 808.35 | 773.90 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 737.95 | 758.82 | 758.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 722.30 | 757.84 | 758.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 679.65 | 675.28 | 701.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 09:15:00 | 654.60 | 675.52 | 701.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-03 09:15:00 | 618.20 | 578.13 | 605.50 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 618.85 | 575.68 | 575.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 625.80 | 576.18 | 575.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 618.70 | 618.78 | 604.96 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 583.20 | 597.96 | 597.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 581.05 | 596.39 | 597.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 10:15:00 | 579.80 | 579.72 | 586.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 11:15:00 | 560.65 | 579.53 | 586.62 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 578.80 | 575.48 | 583.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-07 13:15:00 | 575.75 | 575.51 | 583.45 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-09 10:15:00 | 593.00 | 575.62 | 583.07 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 11:15:00 | 596.45 | 585.65 | 585.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 13:15:00 | 600.95 | 585.91 | 585.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 582.80 | 586.16 | 585.87 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 12:15:00 | 574.65 | 585.49 | 585.54 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 604.00 | 585.53 | 585.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 608.40 | 585.92 | 585.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 12:15:00 | 569.25 | 592.04 | 588.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-04 09:15:00 | 612.95 | 588.42 | 587.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 589.40 | 591.58 | 589.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-09 14:15:00 | 587.65 | 591.51 | 589.26 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 557.30 | 588.88 | 589.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 555.35 | 588.24 | 588.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 528.55 | 528.50 | 543.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 12:15:00 | 523.35 | 528.26 | 541.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 535.25 | 526.16 | 537.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-12 14:15:00 | 539.85 | 526.51 | 537.43 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-15 14:15:00 | 534.80 | 2024-01-17 11:15:00 | 515.00 | EXIT_EMA400 | -19.80 |
| SELL | 2024-04-19 09:15:00 | 433.60 | 2024-04-25 11:15:00 | 458.65 | EXIT_EMA400 | -25.05 |
| BUY | 2024-08-06 11:15:00 | 780.95 | 2024-09-16 09:15:00 | 746.50 | EXIT_EMA400 | -34.45 |
| SELL | 2024-11-26 09:15:00 | 654.60 | 2025-02-03 09:15:00 | 618.20 | EXIT_EMA400 | 36.40 |
| SELL | 2025-07-01 11:15:00 | 560.65 | 2025-07-09 10:15:00 | 593.00 | EXIT_EMA400 | -32.35 |
| SELL | 2025-07-07 13:15:00 | 575.75 | 2025-07-09 10:15:00 | 593.00 | EXIT_EMA400 | -17.25 |
| BUY | 2025-09-04 09:15:00 | 612.95 | 2025-09-09 14:15:00 | 587.65 | EXIT_EMA400 | -25.30 |
| SELL | 2025-12-02 12:15:00 | 523.35 | 2025-12-12 14:15:00 | 539.85 | EXIT_EMA400 | -16.50 |
