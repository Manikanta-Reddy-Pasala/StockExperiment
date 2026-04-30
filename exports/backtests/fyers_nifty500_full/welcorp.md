# Welspun Corp Ltd. (WELCORP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1265.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -130.85
- **Avg P&L per closed trade:** -18.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 11:15:00 | 726.50 | 756.44 | 756.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 712.00 | 755.02 | 755.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 768.10 | 754.42 | 755.44 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 09:15:00 | 785.70 | 756.67 | 756.55 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 13:15:00 | 705.85 | 757.00 | 757.08 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 763.00 | 756.91 | 756.90 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 749.85 | 756.83 | 756.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 734.65 | 756.34 | 756.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 750.75 | 748.00 | 752.17 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 14:15:00 | 793.50 | 755.86 | 755.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 10:15:00 | 805.80 | 756.98 | 756.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 802.60 | 814.83 | 792.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-04 15:15:00 | 819.70 | 814.33 | 792.94 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 784.00 | 814.03 | 792.89 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 13:15:00 | 759.35 | 782.90 | 782.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 749.90 | 781.46 | 782.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 11:15:00 | 775.00 | 775.00 | 778.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-21 11:15:00 | 767.60 | 777.13 | 779.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 776.30 | 775.21 | 777.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-27 09:15:00 | 780.60 | 775.28 | 777.93 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 897.80 | 781.14 | 780.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 901.60 | 786.55 | 783.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 907.00 | 910.30 | 878.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 09:15:00 | 914.60 | 910.34 | 878.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 892.50 | 912.59 | 889.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 15:15:00 | 884.00 | 912.30 | 889.16 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 831.75 | 883.52 | 883.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 10:15:00 | 828.40 | 882.97 | 883.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 873.95 | 872.86 | 877.70 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 894.90 | 881.17 | 881.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 905.95 | 881.74 | 881.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 13:15:00 | 882.10 | 882.57 | 881.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-24 09:15:00 | 885.30 | 882.61 | 881.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 885.30 | 882.61 | 881.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-24 11:15:00 | 881.20 | 882.59 | 881.88 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 857.40 | 881.10 | 881.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 845.15 | 879.54 | 880.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 876.80 | 873.21 | 876.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-09 09:15:00 | 850.40 | 872.06 | 876.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 845.40 | 855.79 | 865.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-23 12:15:00 | 841.85 | 855.45 | 865.48 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 872.70 | 853.84 | 864.12 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 967.70 | 872.46 | 872.34 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 845.55 | 880.35 | 880.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 837.45 | 879.92 | 880.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 829.50 | 826.09 | 845.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-30 09:15:00 | 805.00 | 825.86 | 844.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 799.30 | 765.71 | 794.47 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 808.35 | 805.07 | 805.06 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 796.25 | 804.96 | 805.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 790.05 | 804.71 | 804.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 806.70 | 804.08 | 804.55 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 818.00 | 805.00 | 805.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 824.00 | 805.31 | 805.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 794.50 | 805.78 | 805.40 | EMA200 retest candle locked |

### Cycle 17 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 767.20 | 804.69 | 804.87 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 10:15:00 | 816.50 | 804.90 | 804.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 09:15:00 | 828.10 | 805.68 | 805.28 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-04-04 15:15:00 | 819.70 | 2025-04-07 09:15:00 | 784.00 | EXIT_EMA400 | -35.70 |
| SELL | 2025-05-21 11:15:00 | 767.60 | 2025-05-27 09:15:00 | 780.60 | EXIT_EMA400 | -13.00 |
| BUY | 2025-07-14 09:15:00 | 914.60 | 2025-07-25 15:15:00 | 884.00 | EXIT_EMA400 | -30.60 |
| BUY | 2025-09-24 09:15:00 | 885.30 | 2025-09-24 11:15:00 | 881.20 | EXIT_EMA400 | -4.10 |
| SELL | 2025-10-09 09:15:00 | 850.40 | 2025-10-27 09:15:00 | 872.70 | EXIT_EMA400 | -22.30 |
| SELL | 2025-10-23 12:15:00 | 841.85 | 2025-10-27 09:15:00 | 872.70 | EXIT_EMA400 | -30.85 |
| SELL | 2025-12-30 09:15:00 | 805.00 | 2026-02-03 09:15:00 | 799.30 | EXIT_EMA400 | 5.70 |
