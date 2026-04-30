# Kec International Ltd. (KEC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 560.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -185.45
- **Avg P&L per closed trade:** -26.49

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 919.65 | 1063.13 | 1063.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 901.10 | 1061.51 | 1062.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 760.50 | 750.69 | 824.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-03 12:15:00 | 729.75 | 768.97 | 813.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-15 09:15:00 | 756.20 | 724.33 | 753.97 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 875.50 | 773.73 | 773.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 881.00 | 799.74 | 787.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 13:15:00 | 880.05 | 881.81 | 851.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-09 09:15:00 | 893.80 | 881.85 | 852.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-23 10:15:00 | 857.05 | 879.79 | 859.91 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 797.55 | 851.03 | 851.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 794.75 | 849.94 | 850.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 844.15 | 841.52 | 846.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-18 14:15:00 | 828.00 | 841.65 | 846.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 828.00 | 841.65 | 846.22 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-20 14:15:00 | 821.25 | 840.11 | 845.12 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 855.00 | 830.51 | 838.63 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 880.80 | 844.63 | 844.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 891.15 | 846.27 | 845.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 857.00 | 861.08 | 854.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 14:15:00 | 874.20 | 858.89 | 853.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-01 09:15:00 | 851.50 | 858.86 | 853.68 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 827.60 | 852.04 | 852.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 826.70 | 851.09 | 851.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 746.10 | 722.10 | 756.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-30 09:15:00 | 726.85 | 726.97 | 753.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 747.10 | 728.77 | 752.22 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-02 11:15:00 | 740.45 | 729.06 | 752.13 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-05 14:15:00 | 753.05 | 730.79 | 751.89 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-03 12:15:00 | 729.75 | 2025-05-15 09:15:00 | 756.20 | EXIT_EMA400 | -26.45 |
| BUY | 2025-07-09 09:15:00 | 893.80 | 2025-07-23 10:15:00 | 857.05 | EXIT_EMA400 | -36.75 |
| SELL | 2025-08-18 14:15:00 | 828.00 | 2025-09-02 09:15:00 | 855.00 | EXIT_EMA400 | -27.00 |
| SELL | 2025-08-20 14:15:00 | 821.25 | 2025-09-02 09:15:00 | 855.00 | EXIT_EMA400 | -33.75 |
| BUY | 2025-09-30 14:15:00 | 874.20 | 2025-10-01 09:15:00 | 851.50 | EXIT_EMA400 | -22.70 |
| SELL | 2025-12-30 09:15:00 | 726.85 | 2026-01-05 14:15:00 | 753.05 | EXIT_EMA400 | -26.20 |
| SELL | 2026-01-02 11:15:00 | 740.45 | 2026-01-05 14:15:00 | 753.05 | EXIT_EMA400 | -12.60 |
