# HDFC Bank (HDFCBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 771.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 75.64
- **Avg P&L per closed trade:** 8.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 10:15:00 | 826.20 | 773.27 | 773.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 826.83 | 775.33 | 774.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 822.83 | 824.57 | 808.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-15 09:15:00 | 832.50 | 824.54 | 809.27 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-17 09:15:00 | 790.62 | 825.64 | 810.88 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 716.92 | 798.86 | 798.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 713.60 | 798.01 | 798.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 724.83 | 724.31 | 745.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-11 09:15:00 | 714.75 | 723.97 | 743.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 733.15 | 723.04 | 735.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-02 09:15:00 | 744.62 | 723.86 | 735.84 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 759.75 | 744.00 | 743.92 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 730.47 | 745.93 | 745.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 09:15:00 | 725.97 | 745.44 | 745.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 743.88 | 740.11 | 742.69 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 754.05 | 744.93 | 744.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 09:15:00 | 755.47 | 745.31 | 745.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 741.83 | 749.76 | 747.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 13:15:00 | 767.95 | 750.08 | 747.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-24 12:15:00 | 797.53 | 813.65 | 797.90 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 10:15:00 | 826.45 | 876.38 | 876.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 11:15:00 | 821.00 | 875.83 | 876.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 12:15:00 | 851.33 | 848.52 | 859.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 839.03 | 848.44 | 858.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-04 12:15:00 | 858.15 | 848.08 | 858.13 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 897.05 | 857.20 | 857.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 898.95 | 857.62 | 857.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 10:15:00 | 876.75 | 878.77 | 869.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 12:15:00 | 890.92 | 878.83 | 870.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 985.10 | 995.54 | 984.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-13 13:15:00 | 991.10 | 995.03 | 984.68 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-22 09:15:00 | 982.65 | 995.35 | 986.65 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 958.75 | 980.49 | 980.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 947.40 | 971.45 | 974.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 968.95 | 965.15 | 970.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-03 09:15:00 | 957.75 | 965.04 | 970.68 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-06 09:15:00 | 971.30 | 965.03 | 970.48 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 1002.50 | 974.02 | 973.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.36 | 974.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.97 | 981.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 10:15:00 | 990.00 | 986.44 | 981.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 09:15:00 | 981.05 | 986.38 | 982.03 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.03 | 989.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.44 | 988.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 950.75 | 948.17 | 963.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 15:15:00 | 947.20 | 948.25 | 962.76 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-15 09:15:00 | 832.50 | 2024-01-17 09:15:00 | 790.62 | EXIT_EMA400 | -41.88 |
| SELL | 2024-03-11 09:15:00 | 714.75 | 2024-04-02 09:15:00 | 744.62 | EXIT_EMA400 | -29.88 |
| BUY | 2024-06-05 13:15:00 | 767.95 | 2024-06-19 13:15:00 | 828.59 | TARGET | 60.64 |
| SELL | 2025-02-03 09:15:00 | 839.03 | 2025-02-04 12:15:00 | 858.15 | EXIT_EMA400 | -19.12 |
| BUY | 2025-04-08 12:15:00 | 890.92 | 2025-04-17 12:15:00 | 953.33 | TARGET | 62.41 |
| BUY | 2025-08-13 13:15:00 | 991.10 | 2025-08-18 09:15:00 | 1010.37 | TARGET | 19.27 |
| SELL | 2025-10-03 09:15:00 | 957.75 | 2025-10-06 09:15:00 | 971.30 | EXIT_EMA400 | -13.55 |
| BUY | 2025-11-10 10:15:00 | 990.00 | 2025-11-11 09:15:00 | 981.05 | EXIT_EMA400 | -8.95 |
| SELL | 2026-02-03 15:15:00 | 947.20 | 2026-02-26 12:15:00 | 900.51 | TARGET | 46.69 |
