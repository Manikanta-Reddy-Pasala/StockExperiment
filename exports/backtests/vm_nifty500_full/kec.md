# Kec International Ltd. (KEC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 560.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -240.45
- **Avg P&L per closed trade:** -26.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 10:15:00 | 590.50 | 642.47 | 642.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 14:15:00 | 583.00 | 640.21 | 641.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 09:15:00 | 604.00 | 600.33 | 614.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-26 10:15:00 | 593.10 | 609.25 | 614.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-01-08 09:15:00 | 616.30 | 603.62 | 610.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 12:15:00 | 635.95 | 613.17 | 613.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 655.45 | 614.11 | 613.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 14:15:00 | 687.20 | 687.85 | 663.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-14 11:15:00 | 694.10 | 688.00 | 664.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-15 14:15:00 | 663.00 | 686.93 | 665.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 919.70 | 1063.21 | 1063.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 901.35 | 1061.60 | 1062.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 760.00 | 751.62 | 826.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-03 12:15:00 | 729.75 | 769.41 | 814.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-15 09:15:00 | 756.20 | 724.44 | 754.51 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 874.50 | 774.85 | 774.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 881.00 | 799.84 | 788.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 13:15:00 | 880.40 | 881.86 | 851.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-09 09:15:00 | 893.80 | 881.91 | 852.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-23 10:15:00 | 857.05 | 879.85 | 860.05 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 12:15:00 | 799.00 | 851.60 | 851.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 794.75 | 849.97 | 850.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 844.40 | 841.54 | 846.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-18 14:15:00 | 828.00 | 841.68 | 846.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 828.00 | 841.68 | 846.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-20 14:15:00 | 821.25 | 840.11 | 845.19 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 855.00 | 830.54 | 838.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 880.80 | 844.69 | 844.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 891.10 | 846.32 | 845.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 857.00 | 861.17 | 854.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 14:15:00 | 874.90 | 858.97 | 853.73 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-01 09:15:00 | 851.50 | 858.94 | 853.76 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 827.60 | 852.07 | 852.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 826.70 | 851.12 | 851.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 746.10 | 722.14 | 756.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-30 09:15:00 | 726.85 | 726.99 | 754.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 747.10 | 728.78 | 752.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-02 11:15:00 | 740.45 | 729.07 | 752.15 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-05 14:15:00 | 753.05 | 730.80 | 751.91 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-26 10:15:00 | 593.10 | 2024-01-08 09:15:00 | 616.30 | EXIT_EMA400 | -23.20 |
| BUY | 2024-03-14 11:15:00 | 694.10 | 2024-03-15 14:15:00 | 663.00 | EXIT_EMA400 | -31.10 |
| SELL | 2025-04-03 12:15:00 | 729.75 | 2025-05-15 09:15:00 | 756.20 | EXIT_EMA400 | -26.45 |
| BUY | 2025-07-09 09:15:00 | 893.80 | 2025-07-23 10:15:00 | 857.05 | EXIT_EMA400 | -36.75 |
| SELL | 2025-08-18 14:15:00 | 828.00 | 2025-09-02 09:15:00 | 855.00 | EXIT_EMA400 | -27.00 |
| SELL | 2025-08-20 14:15:00 | 821.25 | 2025-09-02 09:15:00 | 855.00 | EXIT_EMA400 | -33.75 |
| BUY | 2025-09-30 14:15:00 | 874.90 | 2025-10-01 09:15:00 | 851.50 | EXIT_EMA400 | -23.40 |
| SELL | 2025-12-30 09:15:00 | 726.85 | 2026-01-05 14:15:00 | 753.05 | EXIT_EMA400 | -26.20 |
| SELL | 2026-01-02 11:15:00 | 740.45 | 2026-01-05 14:15:00 | 753.05 | EXIT_EMA400 | -12.60 |
