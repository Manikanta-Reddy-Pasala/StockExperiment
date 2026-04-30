# Indegene Ltd. (INDGN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 501.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** -17.93
- **Avg P&L per closed trade:** -2.24

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 596.90 | 650.22 | 650.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 09:15:00 | 587.90 | 637.69 | 643.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 616.80 | 612.67 | 625.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 09:15:00 | 606.70 | 615.36 | 625.11 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 623.50 | 614.79 | 624.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-14 13:15:00 | 626.65 | 614.90 | 624.35 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 659.25 | 631.59 | 631.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 11:15:00 | 661.90 | 631.90 | 631.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 628.25 | 632.98 | 632.19 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 597.95 | 631.12 | 631.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 576.35 | 617.12 | 622.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 573.45 | 560.42 | 584.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-12 09:15:00 | 537.45 | 560.31 | 584.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 567.80 | 548.37 | 569.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-28 09:15:00 | 577.20 | 548.82 | 569.66 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 595.65 | 569.35 | 569.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 603.70 | 570.50 | 569.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 581.65 | 588.11 | 580.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-11 11:15:00 | 599.70 | 587.73 | 581.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 586.30 | 589.80 | 583.01 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-17 14:15:00 | 580.65 | 589.67 | 583.34 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 567.20 | 579.75 | 579.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 565.85 | 578.97 | 579.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 13:15:00 | 577.35 | 574.76 | 576.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-18 11:15:00 | 572.00 | 575.22 | 577.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 576.05 | 575.20 | 577.01 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-21 09:15:00 | 566.45 | 575.11 | 576.94 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-01 09:15:00 | 572.25 | 564.17 | 570.22 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 578.55 | 569.42 | 569.40 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 563.40 | 569.45 | 569.47 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 579.40 | 569.54 | 569.51 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 556.05 | 569.46 | 569.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 554.90 | 569.07 | 569.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 14:15:00 | 557.85 | 557.51 | 562.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-23 09:15:00 | 549.75 | 557.34 | 562.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 536.85 | 527.95 | 537.46 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-08 12:15:00 | 521.55 | 528.00 | 537.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 532.95 | 528.44 | 536.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-12 14:15:00 | 526.50 | 528.62 | 536.36 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-18 14:15:00 | 535.90 | 528.34 | 535.20 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 499.15 | 480.90 | 480.83 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-13 09:15:00 | 606.70 | 2025-01-14 13:15:00 | 626.65 | EXIT_EMA400 | -19.95 |
| SELL | 2025-03-12 09:15:00 | 537.45 | 2025-03-28 09:15:00 | 577.20 | EXIT_EMA400 | -39.75 |
| BUY | 2025-06-11 11:15:00 | 599.70 | 2025-06-17 14:15:00 | 580.65 | EXIT_EMA400 | -19.05 |
| SELL | 2025-07-18 11:15:00 | 572.00 | 2025-07-22 09:15:00 | 556.90 | TARGET | 15.10 |
| SELL | 2025-07-21 09:15:00 | 566.45 | 2025-07-31 09:15:00 | 534.99 | TARGET | 31.46 |
| SELL | 2025-10-23 09:15:00 | 549.75 | 2025-11-07 09:15:00 | 511.74 | TARGET | 38.01 |
| SELL | 2025-12-08 12:15:00 | 521.55 | 2025-12-18 14:15:00 | 535.90 | EXIT_EMA400 | -14.35 |
| SELL | 2025-12-12 14:15:00 | 526.50 | 2025-12-18 14:15:00 | 535.90 | EXIT_EMA400 | -9.40 |
