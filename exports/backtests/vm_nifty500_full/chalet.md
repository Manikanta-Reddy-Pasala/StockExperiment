# Chalet Hotels Ltd. (CHALET.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 757.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -123.65
- **Avg P&L per closed trade:** -30.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 768.00 | 811.96 | 812.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 726.25 | 803.41 | 807.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 801.75 | 797.06 | 803.72 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 10:15:00 | 846.75 | 809.39 | 809.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 857.25 | 815.79 | 813.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 828.90 | 830.34 | 822.38 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 13:15:00 | 805.20 | 815.98 | 816.01 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 848.95 | 816.33 | 816.18 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 795.85 | 817.50 | 817.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 13:15:00 | 788.80 | 816.48 | 817.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 801.75 | 799.11 | 806.68 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 875.00 | 812.54 | 812.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 881.45 | 813.23 | 812.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 849.45 | 851.92 | 836.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-19 13:15:00 | 874.40 | 852.27 | 836.56 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 10:15:00 | 848.45 | 867.73 | 850.85 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 784.10 | 904.43 | 904.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 761.10 | 883.21 | 893.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 754.25 | 747.56 | 794.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-03 09:15:00 | 721.70 | 747.31 | 794.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-07 14:15:00 | 790.05 | 751.18 | 788.92 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 871.05 | 804.05 | 803.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 13:15:00 | 878.50 | 817.62 | 812.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 887.95 | 893.14 | 866.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 15:15:00 | 897.90 | 891.88 | 868.83 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-07 12:15:00 | 876.35 | 897.28 | 880.98 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 924.10 | 960.32 | 960.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 909.75 | 958.87 | 959.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 908.75 | 908.43 | 926.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-04 09:15:00 | 902.20 | 908.37 | 925.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 896.05 | 884.84 | 901.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 910.00 | 885.09 | 902.01 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-19 13:15:00 | 874.40 | 2024-10-07 10:15:00 | 848.45 | EXIT_EMA400 | -25.95 |
| SELL | 2025-03-03 09:15:00 | 721.70 | 2025-03-07 14:15:00 | 790.05 | EXIT_EMA400 | -68.35 |
| BUY | 2025-06-17 15:15:00 | 897.90 | 2025-07-07 12:15:00 | 876.35 | EXIT_EMA400 | -21.55 |
| SELL | 2025-12-04 09:15:00 | 902.20 | 2026-01-02 09:15:00 | 910.00 | EXIT_EMA400 | -7.80 |
