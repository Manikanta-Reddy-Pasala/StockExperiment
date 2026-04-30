# Ambuja Cements Ltd. (AMBUJACEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 443.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -2.11
- **Avg P&L per closed trade:** -0.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 624.45 | 645.38 | 645.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 623.70 | 645.16 | 645.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 13:15:00 | 636.00 | 635.38 | 639.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-05 14:15:00 | 632.20 | 635.35 | 639.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 631.80 | 625.69 | 632.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-27 11:15:00 | 632.15 | 625.81 | 632.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 540.85 | 516.33 | 516.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 545.05 | 516.61 | 516.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 535.85 | 540.81 | 531.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-16 09:15:00 | 557.30 | 539.09 | 533.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 547.00 | 553.96 | 546.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 543.40 | 553.86 | 546.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 552.25 | 577.87 | 577.92 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 590.50 | 577.72 | 577.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 565.00 | 577.78 | 577.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 560.00 | 575.73 | 576.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 570.30 | 568.28 | 571.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 13:15:00 | 565.10 | 568.30 | 571.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 571.20 | 568.16 | 571.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-03 12:15:00 | 575.70 | 568.26 | 571.50 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-05 14:15:00 | 632.20 | 2024-09-19 11:15:00 | 609.81 | TARGET | 22.39 |
| BUY | 2025-05-16 09:15:00 | 557.30 | 2025-06-13 09:15:00 | 543.40 | EXIT_EMA400 | -13.90 |
| SELL | 2025-10-30 13:15:00 | 565.10 | 2025-11-03 12:15:00 | 575.70 | EXIT_EMA400 | -10.60 |
