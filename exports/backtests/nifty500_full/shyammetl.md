# Shyam Metalics and Energy Ltd. (SHYAMMETL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 873.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 13 |
| ALERT2 | 12 |
| ALERT3 | 3 |
| ENTRY1 | 9 |
| ENTRY2 | 1 |
| EXIT | 9 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 91.40
- **Avg P&L per closed trade:** 9.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 10:15:00 | 599.80 | 615.59 | 615.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 594.50 | 613.79 | 614.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 11:15:00 | 618.40 | 613.57 | 614.56 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 11:15:00 | 626.00 | 615.42 | 615.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 09:15:00 | 631.10 | 616.33 | 615.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 602.10 | 617.39 | 616.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 11:15:00 | 608.35 | 617.15 | 616.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 608.35 | 617.15 | 616.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-15 12:15:00 | 606.95 | 617.05 | 616.31 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 15:15:00 | 597.10 | 615.49 | 615.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 10:15:00 | 592.90 | 614.60 | 615.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 613.70 | 612.88 | 614.13 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 12:15:00 | 645.90 | 615.48 | 615.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 13:15:00 | 649.50 | 615.82 | 615.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 10:15:00 | 620.50 | 620.54 | 618.20 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 10:15:00 | 586.60 | 615.96 | 616.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 578.10 | 614.04 | 615.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 15:15:00 | 607.50 | 606.86 | 611.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-16 11:15:00 | 594.95 | 606.40 | 610.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-17 10:15:00 | 622.75 | 605.90 | 610.27 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 15:15:00 | 647.05 | 614.03 | 614.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 647.55 | 614.36 | 614.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 10:15:00 | 617.45 | 617.61 | 615.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-29 13:15:00 | 626.80 | 617.76 | 616.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-30 09:15:00 | 614.90 | 617.85 | 616.12 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 552.30 | 614.61 | 614.70 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 10:15:00 | 642.90 | 614.48 | 614.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 645.95 | 616.06 | 615.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 655.70 | 675.47 | 655.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-23 13:15:00 | 681.20 | 674.14 | 656.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-22 09:15:00 | 839.30 | 887.09 | 841.47 | Close below EMA400 |

### Cycle 9 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 790.85 | 832.06 | 832.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 786.00 | 830.90 | 831.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 789.00 | 786.12 | 803.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-08 15:15:00 | 781.00 | 786.07 | 803.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 779.25 | 785.54 | 802.69 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-20 11:15:00 | 801.10 | 777.37 | 794.74 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 12:15:00 | 855.00 | 771.12 | 770.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 13:15:00 | 859.30 | 771.99 | 771.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 791.80 | 825.38 | 803.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 818.00 | 824.16 | 803.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 866.90 | 885.44 | 860.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-27 10:15:00 | 872.00 | 885.31 | 860.87 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-29 10:15:00 | 860.10 | 883.30 | 861.48 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 839.00 | 854.31 | 854.32 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 866.20 | 854.35 | 854.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 10:15:00 | 880.35 | 857.17 | 855.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 861.90 | 861.99 | 858.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 15:15:00 | 871.95 | 860.52 | 858.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 14:15:00 | 907.50 | 940.49 | 919.14 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 906.10 | 919.98 | 920.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 899.45 | 919.21 | 919.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 835.45 | 822.52 | 848.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 816.55 | 829.42 | 844.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-14 15:15:00 | 840.00 | 824.25 | 838.88 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 12:15:00 | 907.40 | 840.32 | 840.12 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 800.50 | 845.01 | 845.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 791.90 | 844.49 | 844.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 809.60 | 808.04 | 822.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 773.50 | 807.39 | 821.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-07 09:15:00 | 832.00 | 800.92 | 815.82 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 885.80 | 825.28 | 825.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 888.00 | 825.90 | 825.47 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-15 11:15:00 | 608.35 | 2024-04-15 12:15:00 | 606.95 | EXIT_EMA400 | -1.40 |
| SELL | 2024-05-16 11:15:00 | 594.95 | 2024-05-17 10:15:00 | 622.75 | EXIT_EMA400 | -27.80 |
| BUY | 2024-05-29 13:15:00 | 626.80 | 2024-05-30 09:15:00 | 614.90 | EXIT_EMA400 | -11.90 |
| BUY | 2024-07-23 13:15:00 | 681.20 | 2024-08-01 09:15:00 | 754.45 | TARGET | 73.25 |
| SELL | 2025-01-08 15:15:00 | 781.00 | 2025-01-10 09:15:00 | 713.05 | TARGET | 67.95 |
| BUY | 2025-04-07 15:15:00 | 818.00 | 2025-04-15 09:15:00 | 862.56 | TARGET | 44.56 |
| BUY | 2025-05-27 10:15:00 | 872.00 | 2025-05-29 10:15:00 | 860.10 | EXIT_EMA400 | -11.90 |
| BUY | 2025-07-15 15:15:00 | 871.95 | 2025-07-17 09:15:00 | 912.55 | TARGET | 40.60 |
| SELL | 2026-01-08 09:15:00 | 816.55 | 2026-01-14 15:15:00 | 840.00 | EXIT_EMA400 | -23.45 |
| SELL | 2026-03-27 09:15:00 | 773.50 | 2026-04-07 09:15:00 | 832.00 | EXIT_EMA400 | -58.50 |
