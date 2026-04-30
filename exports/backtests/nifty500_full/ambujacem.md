# Ambuja Cements Ltd. (AMBUJACEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 444.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 146.69
- **Avg P&L per closed trade:** 24.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 420.85 | 441.24 | 441.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 15:15:00 | 418.20 | 440.00 | 440.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 435.30 | 435.07 | 437.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-09 12:15:00 | 429.70 | 434.93 | 437.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 436.75 | 434.80 | 437.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-10-11 09:15:00 | 441.55 | 434.93 | 437.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 12:15:00 | 504.50 | 431.48 | 431.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 13:15:00 | 509.30 | 432.25 | 431.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-24 11:15:00 | 515.55 | 517.49 | 495.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-24 14:15:00 | 526.60 | 517.67 | 496.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 568.95 | 584.66 | 558.86 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-14 10:15:00 | 580.85 | 583.74 | 559.03 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-05-07 10:15:00 | 596.75 | 616.73 | 599.42 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 15:15:00 | 631.80 | 648.34 | 648.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 627.90 | 647.21 | 647.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 13:15:00 | 636.15 | 635.47 | 640.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-05 14:15:00 | 632.20 | 635.44 | 640.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 631.80 | 625.73 | 632.73 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-27 13:15:00 | 633.50 | 625.99 | 632.72 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 548.80 | 517.06 | 516.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 549.10 | 517.38 | 517.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 535.85 | 540.87 | 531.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-16 09:15:00 | 557.30 | 539.14 | 533.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 547.95 | 554.00 | 546.48 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 543.40 | 553.89 | 546.46 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 552.25 | 577.84 | 577.92 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 590.70 | 577.69 | 577.67 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 565.00 | 577.74 | 577.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 560.00 | 575.71 | 576.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 570.30 | 568.28 | 571.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 13:15:00 | 565.10 | 568.29 | 571.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 571.20 | 568.16 | 571.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-03 12:15:00 | 575.55 | 568.26 | 571.50 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-09 12:15:00 | 429.70 | 2023-10-11 09:15:00 | 441.55 | EXIT_EMA400 | -11.85 |
| BUY | 2024-01-24 14:15:00 | 526.60 | 2024-03-01 13:15:00 | 618.11 | TARGET | 91.51 |
| BUY | 2024-03-14 10:15:00 | 580.85 | 2024-04-24 13:15:00 | 646.32 | TARGET | 65.47 |
| SELL | 2024-09-05 14:15:00 | 632.20 | 2024-09-19 11:15:00 | 606.29 | TARGET | 25.91 |
| BUY | 2025-05-16 09:15:00 | 557.30 | 2025-06-13 09:15:00 | 543.40 | EXIT_EMA400 | -13.90 |
| SELL | 2025-10-30 13:15:00 | 565.10 | 2025-11-03 12:15:00 | 575.55 | EXIT_EMA400 | -10.45 |
