# Elecon Engineering Co. Ltd. (ELECON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 507.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 43.26
- **Avg P&L per closed trade:** 8.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 13:15:00 | 542.50 | 632.77 | 632.91 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 648.45 | 610.95 | 610.92 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 551.35 | 614.05 | 614.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 15:15:00 | 549.00 | 613.41 | 613.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 14:15:00 | 617.90 | 607.97 | 611.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 12:15:00 | 577.50 | 605.42 | 609.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 473.80 | 443.92 | 484.36 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 443.40 | 444.41 | 483.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 466.30 | 440.82 | 467.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-16 10:15:00 | 475.00 | 441.16 | 467.49 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 567.35 | 484.39 | 484.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 13:15:00 | 575.60 | 485.30 | 484.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 653.05 | 654.45 | 607.72 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 578.90 | 614.33 | 614.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 570.90 | 613.55 | 613.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 581.05 | 580.80 | 593.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 13:15:00 | 566.45 | 580.16 | 592.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 10:15:00 | 592.30 | 567.91 | 580.78 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 605.00 | 589.65 | 589.60 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 575.35 | 589.46 | 589.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 569.50 | 589.26 | 589.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 587.70 | 585.08 | 587.21 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 607.40 | 589.00 | 588.92 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 552.20 | 588.66 | 588.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 537.05 | 585.64 | 587.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 569.50 | 569.13 | 576.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 12:15:00 | 563.85 | 568.98 | 576.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 508.60 | 491.24 | 511.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-06 11:15:00 | 484.00 | 491.95 | 510.77 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-07 10:15:00 | 515.90 | 491.83 | 510.15 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 509.45 | 424.66 | 424.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 510.05 | 426.33 | 425.48 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-22 12:15:00 | 577.50 | 2025-02-03 13:15:00 | 482.46 | TARGET | 95.04 |
| SELL | 2025-03-25 10:15:00 | 443.40 | 2025-04-16 10:15:00 | 475.00 | EXIT_EMA400 | -31.60 |
| SELL | 2025-08-21 13:15:00 | 566.45 | 2025-09-10 10:15:00 | 592.30 | EXIT_EMA400 | -25.85 |
| SELL | 2025-10-31 12:15:00 | 563.85 | 2025-11-10 09:15:00 | 526.29 | TARGET | 37.56 |
| SELL | 2026-01-06 11:15:00 | 484.00 | 2026-01-07 10:15:00 | 515.90 | EXIT_EMA400 | -31.90 |
