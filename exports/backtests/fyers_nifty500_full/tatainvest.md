# Tata Investment Corporation Ltd. (TATAINVEST.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 719.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -155.89
- **Avg P&L per closed trade:** -38.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 14:15:00 | 714.15 | 640.78 | 640.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 753.34 | 642.64 | 641.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 678.84 | 680.95 | 666.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-23 10:15:00 | 709.92 | 679.60 | 666.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-03 09:15:00 | 670.98 | 683.04 | 671.62 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 649.13 | 673.50 | 673.60 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 689.62 | 673.43 | 673.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 696.00 | 677.50 | 675.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 10:15:00 | 679.20 | 679.91 | 676.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 14:15:00 | 689.00 | 680.10 | 677.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-17 12:15:00 | 677.30 | 680.71 | 677.59 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 645.80 | 676.42 | 676.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 638.60 | 675.40 | 675.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 594.75 | 588.77 | 617.55 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 11:15:00 | 630.20 | 619.07 | 619.03 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 10:15:00 | 611.85 | 619.04 | 619.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 608.50 | 618.73 | 618.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 607.50 | 607.14 | 612.26 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 15:15:00 | 630.00 | 615.39 | 615.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 635.85 | 615.60 | 615.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 654.50 | 657.22 | 641.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 14:15:00 | 661.65 | 656.39 | 642.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 658.80 | 667.20 | 655.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 15:15:00 | 654.00 | 666.53 | 655.30 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 758.25 | 788.33 | 788.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 748.00 | 787.64 | 788.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 654.55 | 648.96 | 682.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 626.60 | 650.78 | 679.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-19 09:15:00 | 724.20 | 647.89 | 674.27 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 729.50 | 650.84 | 650.72 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-23 10:15:00 | 709.92 | 2024-10-03 09:15:00 | 670.98 | EXIT_EMA400 | -38.94 |
| BUY | 2024-12-13 14:15:00 | 689.00 | 2024-12-17 12:15:00 | 677.30 | EXIT_EMA400 | -11.70 |
| BUY | 2025-06-20 14:15:00 | 661.65 | 2025-07-14 15:15:00 | 654.00 | EXIT_EMA400 | -7.65 |
| SELL | 2026-02-13 09:15:00 | 626.60 | 2026-02-19 09:15:00 | 724.20 | EXIT_EMA400 | -97.60 |
