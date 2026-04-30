# Indegene Ltd. (INDGN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:30:00 (3387 bars)
- **Last close:** 499.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -65.19
- **Avg P&L per closed trade:** -10.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 14:15:00 | 591.05 | 649.77 | 650.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 15:15:00 | 589.55 | 649.17 | 649.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 616.80 | 612.67 | 625.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 09:15:00 | 606.45 | 615.34 | 625.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 623.50 | 614.77 | 624.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-14 13:15:00 | 626.65 | 614.88 | 624.35 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 659.20 | 631.59 | 631.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 11:15:00 | 661.90 | 631.89 | 631.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 627.85 | 632.98 | 632.19 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 597.95 | 631.11 | 631.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 576.30 | 617.80 | 623.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 573.25 | 560.65 | 585.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-12 09:15:00 | 537.90 | 560.55 | 584.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 567.80 | 548.47 | 569.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-28 09:15:00 | 577.20 | 548.94 | 569.88 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 13:15:00 | 595.00 | 569.58 | 569.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 15:15:00 | 598.50 | 570.13 | 569.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 581.85 | 588.08 | 580.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-11 11:15:00 | 599.70 | 587.66 | 581.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 585.75 | 589.74 | 583.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-17 14:15:00 | 580.65 | 589.62 | 583.33 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 567.20 | 579.78 | 579.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 565.85 | 579.00 | 579.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 13:15:00 | 577.35 | 574.77 | 577.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-18 11:15:00 | 572.00 | 575.23 | 577.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 576.05 | 575.22 | 577.03 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-21 09:15:00 | 566.45 | 575.13 | 576.96 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-01 09:15:00 | 572.70 | 564.23 | 570.27 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 578.55 | 569.43 | 569.41 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 563.40 | 569.46 | 569.48 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 579.75 | 569.54 | 569.52 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 556.05 | 569.46 | 569.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 555.00 | 569.07 | 569.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 12:15:00 | 554.75 | 553.93 | 559.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 09:15:00 | 540.35 | 553.71 | 559.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-18 14:15:00 | 535.90 | 528.33 | 535.19 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 12:15:00 | 502.80 | 481.19 | 481.14 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-13 09:15:00 | 606.45 | 2025-01-14 13:15:00 | 626.65 | EXIT_EMA400 | -20.20 |
| SELL | 2025-03-12 09:15:00 | 537.90 | 2025-03-28 09:15:00 | 577.20 | EXIT_EMA400 | -39.30 |
| BUY | 2025-06-11 11:15:00 | 599.70 | 2025-06-17 14:15:00 | 580.65 | EXIT_EMA400 | -19.05 |
| SELL | 2025-07-18 11:15:00 | 572.00 | 2025-07-22 09:15:00 | 556.84 | TARGET | 15.16 |
| SELL | 2025-07-21 09:15:00 | 566.45 | 2025-08-01 09:15:00 | 572.70 | EXIT_EMA400 | -6.25 |
| SELL | 2025-10-31 09:15:00 | 540.35 | 2025-12-18 14:15:00 | 535.90 | EXIT_EMA400 | 4.45 |
