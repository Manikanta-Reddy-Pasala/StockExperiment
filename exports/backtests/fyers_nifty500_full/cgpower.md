# CG Power and Industrial Solutions Ltd. (CGPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 811.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -156.70
- **Avg P&L per closed trade:** -31.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 709.85 | 736.25 | 736.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 700.00 | 735.01 | 735.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 15:15:00 | 732.10 | 731.69 | 733.87 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 762.30 | 735.91 | 735.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 777.60 | 740.73 | 738.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 754.35 | 757.78 | 749.02 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 717.35 | 744.33 | 744.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 15:15:00 | 717.00 | 744.05 | 744.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 611.40 | 610.83 | 645.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-27 09:15:00 | 586.80 | 609.93 | 644.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 619.70 | 606.62 | 634.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 10:15:00 | 613.75 | 606.69 | 634.50 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-18 14:15:00 | 633.10 | 607.26 | 629.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 667.75 | 625.75 | 625.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 678.30 | 627.48 | 626.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 15:15:00 | 673.55 | 674.09 | 658.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 10:15:00 | 679.50 | 673.85 | 659.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 667.65 | 677.19 | 666.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-02 14:15:00 | 665.80 | 676.89 | 666.33 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 689.50 | 728.25 | 728.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 14:15:00 | 688.60 | 727.48 | 728.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 613.35 | 607.14 | 639.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 11:15:00 | 593.65 | 607.01 | 639.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 653.00 | 606.59 | 637.32 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 713.40 | 655.22 | 655.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 719.20 | 657.55 | 656.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 10:15:00 | 684.20 | 692.81 | 678.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 14:15:00 | 698.45 | 692.75 | 678.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-20 14:15:00 | 680.45 | 694.14 | 681.50 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-27 09:15:00 | 586.80 | 2025-03-18 14:15:00 | 633.10 | EXIT_EMA400 | -46.30 |
| SELL | 2025-03-10 10:15:00 | 613.75 | 2025-03-18 14:15:00 | 633.10 | EXIT_EMA400 | -19.35 |
| BUY | 2025-06-16 10:15:00 | 679.50 | 2025-07-02 14:15:00 | 665.80 | EXIT_EMA400 | -13.70 |
| SELL | 2026-02-01 11:15:00 | 593.65 | 2026-02-03 09:15:00 | 653.00 | EXIT_EMA400 | -59.35 |
| BUY | 2026-03-16 14:15:00 | 698.45 | 2026-03-20 14:15:00 | 680.45 | EXIT_EMA400 | -18.00 |
