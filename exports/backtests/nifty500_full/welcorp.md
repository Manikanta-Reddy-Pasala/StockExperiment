# Welspun Corp Ltd. (WELCORP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1266.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 20 |
| ALERT1 | 15 |
| ALERT2 | 14 |
| ALERT3 | 4 |
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| EXIT | 9 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -66.12
- **Avg P&L per closed trade:** -7.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 13:15:00 | 506.80 | 538.88 | 538.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 10:15:00 | 500.50 | 537.74 | 538.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 09:15:00 | 537.10 | 533.16 | 535.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-26 12:15:00 | 520.90 | 533.07 | 535.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-01 09:15:00 | 539.90 | 530.77 | 534.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 14:15:00 | 582.10 | 537.68 | 537.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 15:15:00 | 585.00 | 538.16 | 537.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 547.60 | 548.56 | 543.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 10:15:00 | 554.15 | 548.61 | 543.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-18 14:15:00 | 536.30 | 549.40 | 544.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 09:15:00 | 528.10 | 561.44 | 561.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 14:15:00 | 524.00 | 559.83 | 560.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 550.15 | 545.72 | 552.42 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 610.20 | 557.04 | 556.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 616.90 | 559.18 | 558.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 616.10 | 623.88 | 601.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-05 15:15:00 | 631.00 | 623.80 | 601.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 664.05 | 683.37 | 661.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-19 12:15:00 | 660.05 | 683.13 | 661.73 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 743.55 | 756.30 | 756.36 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 12:15:00 | 800.50 | 756.52 | 756.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 13:15:00 | 804.55 | 757.00 | 756.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 763.20 | 763.87 | 760.46 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 15:15:00 | 709.00 | 756.94 | 757.16 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 13:15:00 | 764.45 | 757.20 | 757.20 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 753.55 | 757.20 | 757.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 734.60 | 756.94 | 757.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 750.75 | 748.48 | 752.58 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 14:15:00 | 793.60 | 756.22 | 756.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 10:15:00 | 805.20 | 757.37 | 756.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 802.60 | 815.01 | 793.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-04 15:15:00 | 819.40 | 814.51 | 793.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 783.60 | 814.21 | 793.13 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 13:15:00 | 759.35 | 782.99 | 783.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 749.90 | 781.50 | 782.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 783.80 | 775.03 | 778.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-21 11:15:00 | 767.60 | 777.20 | 779.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 776.15 | 775.31 | 778.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-27 09:15:00 | 780.60 | 775.38 | 778.06 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 898.00 | 781.22 | 780.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 901.60 | 786.62 | 783.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 907.65 | 910.78 | 879.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-17 09:15:00 | 930.85 | 910.52 | 881.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 892.50 | 912.66 | 889.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 15:15:00 | 884.00 | 912.38 | 889.23 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 832.00 | 883.56 | 883.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 10:15:00 | 828.40 | 883.01 | 883.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 874.30 | 872.89 | 877.75 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 894.75 | 881.21 | 881.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 905.95 | 881.77 | 881.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 13:15:00 | 882.10 | 882.64 | 881.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-24 09:15:00 | 884.70 | 882.67 | 881.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 884.70 | 882.67 | 881.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-24 11:15:00 | 880.60 | 882.65 | 881.93 | Close below EMA400 |

### Cycle 15 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 857.40 | 881.15 | 881.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 855.95 | 880.40 | 880.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 877.25 | 873.23 | 876.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-09 09:15:00 | 850.40 | 872.07 | 876.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 872.70 | 853.82 | 864.13 | Close above EMA400 |

### Cycle 16 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 967.70 | 872.47 | 872.36 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 845.55 | 880.38 | 880.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 837.45 | 879.96 | 880.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 829.25 | 826.15 | 845.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-30 09:15:00 | 805.00 | 825.92 | 844.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 799.30 | 767.79 | 796.54 | Close above EMA400 |

### Cycle 18 — BUY (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 12:15:00 | 811.00 | 806.24 | 806.22 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 794.50 | 806.13 | 806.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 771.20 | 805.42 | 805.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 13:15:00 | 805.15 | 802.73 | 804.38 | EMA200 retest candle locked |

### Cycle 20 — BUY (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 09:15:00 | 828.00 | 806.00 | 805.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 850.85 | 811.65 | 808.93 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-26 12:15:00 | 520.90 | 2024-04-01 09:15:00 | 539.90 | EXIT_EMA400 | -19.00 |
| BUY | 2024-04-15 10:15:00 | 554.15 | 2024-04-18 14:15:00 | 536.30 | EXIT_EMA400 | -17.85 |
| BUY | 2024-08-05 15:15:00 | 631.00 | 2024-08-19 09:15:00 | 718.08 | TARGET | 87.08 |
| BUY | 2025-04-04 15:15:00 | 819.40 | 2025-04-07 09:15:00 | 783.60 | EXIT_EMA400 | -35.80 |
| SELL | 2025-05-21 11:15:00 | 767.60 | 2025-05-27 09:15:00 | 780.60 | EXIT_EMA400 | -13.00 |
| BUY | 2025-07-17 09:15:00 | 930.85 | 2025-07-25 15:15:00 | 884.00 | EXIT_EMA400 | -46.85 |
| BUY | 2025-09-24 09:15:00 | 884.70 | 2025-09-24 11:15:00 | 880.60 | EXIT_EMA400 | -4.10 |
| SELL | 2025-10-09 09:15:00 | 850.40 | 2025-10-27 09:15:00 | 872.70 | EXIT_EMA400 | -22.30 |
| SELL | 2025-12-30 09:15:00 | 805.00 | 2026-02-03 09:15:00 | 799.30 | EXIT_EMA400 | 5.70 |
