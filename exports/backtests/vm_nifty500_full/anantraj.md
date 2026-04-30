# Anant Raj Ltd. (ANANTRAJ.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 486.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -69.30
- **Avg P&L per closed trade:** -17.32

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 594.35 | 761.66 | 762.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 13:15:00 | 573.40 | 748.10 | 755.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 546.45 | 535.71 | 596.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 09:15:00 | 516.50 | 535.36 | 590.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-16 10:15:00 | 507.00 | 468.71 | 504.10 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 575.50 | 520.43 | 520.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 592.55 | 522.22 | 521.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 531.75 | 535.13 | 528.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 14:15:00 | 553.30 | 535.55 | 528.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 538.80 | 536.10 | 529.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 13:15:00 | 527.55 | 535.91 | 529.57 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 520.00 | 548.88 | 548.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 512.40 | 548.52 | 548.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 545.85 | 544.23 | 546.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 12:15:00 | 535.80 | 543.86 | 546.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 566.50 | 539.26 | 543.14 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 587.60 | 546.98 | 546.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 591.70 | 548.97 | 547.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 639.65 | 648.63 | 613.56 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 563.80 | 614.10 | 614.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 559.40 | 613.06 | 613.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 564.50 | 562.69 | 580.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 13:15:00 | 553.20 | 562.69 | 579.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 13:15:00 | 575.55 | 558.37 | 574.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-26 09:15:00 | 516.50 | 2025-05-16 10:15:00 | 507.00 | EXIT_EMA400 | 9.50 |
| BUY | 2025-06-16 14:15:00 | 553.30 | 2025-06-18 13:15:00 | 527.55 | EXIT_EMA400 | -25.75 |
| SELL | 2025-09-04 12:15:00 | 535.80 | 2025-09-15 09:15:00 | 566.50 | EXIT_EMA400 | -30.70 |
| SELL | 2025-12-26 13:15:00 | 553.20 | 2026-01-02 13:15:00 | 575.55 | EXIT_EMA400 | -22.35 |
