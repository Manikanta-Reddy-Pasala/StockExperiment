# UPL Ltd. (UPL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 641.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 23.47
- **Avg P&L per closed trade:** 3.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 512.35 | 498.25 | 498.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 519.95 | 499.06 | 498.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 494.90 | 506.49 | 502.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 10:15:00 | 522.20 | 506.31 | 502.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 542.80 | 555.58 | 541.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-22 09:15:00 | 548.40 | 555.51 | 541.60 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 542.15 | 555.01 | 541.82 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-23 10:15:00 | 552.30 | 554.99 | 541.87 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 542.60 | 554.86 | 541.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-23 12:15:00 | 535.35 | 554.67 | 541.84 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 532.25 | 576.66 | 576.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 522.90 | 573.28 | 575.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 562.75 | 561.57 | 567.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 555.10 | 562.02 | 567.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-21 12:15:00 | 560.95 | 551.49 | 560.49 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 607.40 | 545.03 | 544.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 626.50 | 546.42 | 545.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 11:15:00 | 611.60 | 612.38 | 590.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-05 09:15:00 | 623.45 | 612.50 | 590.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 606.55 | 634.27 | 615.96 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 650.95 | 688.13 | 688.14 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 723.45 | 685.70 | 685.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 736.25 | 740.97 | 724.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-10 09:15:00 | 748.00 | 740.93 | 725.23 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 763.70 | 775.31 | 757.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-20 10:15:00 | 755.75 | 775.12 | 757.93 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 678.40 | 744.68 | 744.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 657.25 | 743.55 | 744.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 633.75 | 629.93 | 662.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 11:15:00 | 628.00 | 629.92 | 662.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 659.50 | 634.07 | 660.20 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 660.70 | 634.59 | 660.19 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-05 10:15:00 | 522.20 | 2024-07-02 09:15:00 | 579.99 | TARGET | 57.79 |
| BUY | 2024-07-22 09:15:00 | 548.40 | 2024-07-23 12:15:00 | 535.35 | EXIT_EMA400 | -13.05 |
| BUY | 2024-07-23 10:15:00 | 552.30 | 2024-07-23 12:15:00 | 535.35 | EXIT_EMA400 | -16.95 |
| SELL | 2024-11-08 15:15:00 | 555.10 | 2024-11-11 14:15:00 | 517.57 | TARGET | 37.53 |
| BUY | 2025-03-05 09:15:00 | 623.45 | 2025-04-07 09:15:00 | 606.55 | EXIT_EMA400 | -16.90 |
| BUY | 2025-12-10 09:15:00 | 748.00 | 2026-01-20 10:15:00 | 755.75 | EXIT_EMA400 | 7.75 |
| SELL | 2026-04-08 11:15:00 | 628.00 | 2026-04-16 09:15:00 | 660.70 | EXIT_EMA400 | -32.70 |
