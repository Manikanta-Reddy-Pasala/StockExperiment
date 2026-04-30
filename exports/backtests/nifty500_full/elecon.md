# Elecon Engineering Co. Ltd. (ELECON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 506.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -31.90
- **Avg P&L per closed trade:** -4.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 09:15:00 | 411.58 | 486.82 | 486.92 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 497.90 | 480.43 | 480.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 09:15:00 | 509.62 | 482.40 | 481.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 524.47 | 528.60 | 509.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-08 12:15:00 | 534.25 | 528.25 | 510.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 511.58 | 527.82 | 511.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-09 14:15:00 | 505.00 | 527.59 | 511.04 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 547.25 | 633.77 | 633.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 13:15:00 | 542.50 | 632.86 | 633.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 596.80 | 587.78 | 604.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 09:15:00 | 570.60 | 587.69 | 604.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 621.50 | 586.49 | 601.10 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 649.35 | 611.42 | 611.40 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 551.35 | 614.13 | 614.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 15:15:00 | 550.00 | 613.50 | 614.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 14:15:00 | 617.90 | 608.02 | 611.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 12:15:00 | 577.50 | 605.46 | 609.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 473.80 | 444.25 | 485.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 443.40 | 444.72 | 484.29 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 466.30 | 440.96 | 467.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-16 10:15:00 | 474.00 | 441.29 | 467.91 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 567.60 | 484.46 | 484.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 13:15:00 | 575.50 | 485.36 | 484.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 653.35 | 654.46 | 607.81 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 578.90 | 614.32 | 614.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 570.90 | 613.54 | 614.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 581.10 | 580.77 | 593.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 13:15:00 | 566.45 | 580.14 | 592.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 10:15:00 | 592.30 | 567.92 | 580.80 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 605.00 | 589.68 | 589.62 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 575.35 | 589.49 | 589.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 569.50 | 589.30 | 589.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 587.70 | 585.09 | 587.22 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 607.40 | 589.00 | 588.93 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 552.20 | 588.65 | 588.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 544.10 | 585.69 | 587.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 569.50 | 569.19 | 576.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 12:15:00 | 563.85 | 569.08 | 576.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 508.60 | 491.19 | 511.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-06 10:15:00 | 487.30 | 491.97 | 510.87 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-07 10:15:00 | 515.80 | 491.78 | 510.12 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 510.05 | 425.79 | 425.63 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-08 12:15:00 | 534.25 | 2024-05-09 14:15:00 | 505.00 | EXIT_EMA400 | -29.25 |
| SELL | 2024-11-26 09:15:00 | 570.60 | 2024-12-03 09:15:00 | 621.50 | EXIT_EMA400 | -50.90 |
| SELL | 2025-01-22 12:15:00 | 577.50 | 2025-02-03 13:15:00 | 482.05 | TARGET | 95.45 |
| SELL | 2025-03-25 10:15:00 | 443.40 | 2025-04-16 10:15:00 | 474.00 | EXIT_EMA400 | -30.60 |
| SELL | 2025-08-21 13:15:00 | 566.45 | 2025-09-10 10:15:00 | 592.30 | EXIT_EMA400 | -25.85 |
| SELL | 2025-10-31 12:15:00 | 563.85 | 2025-11-10 09:15:00 | 526.10 | TARGET | 37.75 |
| SELL | 2026-01-06 10:15:00 | 487.30 | 2026-01-07 10:15:00 | 515.80 | EXIT_EMA400 | -28.50 |
