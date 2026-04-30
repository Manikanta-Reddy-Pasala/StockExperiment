# Shyam Metalics and Energy Ltd. (SHYAMMETL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 870.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 50.68
- **Avg P&L per closed trade:** 8.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 790.85 | 831.99 | 832.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 786.00 | 830.83 | 831.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 789.00 | 786.19 | 803.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-09 10:15:00 | 776.30 | 786.03 | 803.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 779.25 | 785.62 | 802.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-20 11:15:00 | 801.10 | 777.43 | 794.70 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 11:15:00 | 850.50 | 770.15 | 770.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 12:15:00 | 855.00 | 770.99 | 770.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 791.80 | 825.16 | 802.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 818.00 | 823.96 | 802.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 866.90 | 885.47 | 860.75 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-27 10:15:00 | 872.00 | 885.34 | 860.80 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-29 10:15:00 | 860.10 | 883.30 | 861.41 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 839.00 | 854.20 | 854.22 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 866.20 | 854.24 | 854.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 10:15:00 | 880.35 | 857.06 | 855.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 862.10 | 862.17 | 858.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 10:15:00 | 864.05 | 860.03 | 858.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 14:15:00 | 907.50 | 940.42 | 919.06 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 906.10 | 919.93 | 919.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 899.45 | 919.15 | 919.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 835.45 | 822.51 | 848.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 816.55 | 829.36 | 843.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-14 15:15:00 | 840.00 | 824.22 | 838.85 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 903.25 | 839.90 | 839.74 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 791.90 | 844.78 | 845.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 765.35 | 843.48 | 844.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 809.60 | 808.03 | 822.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 773.50 | 807.39 | 821.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-07 09:15:00 | 832.00 | 800.88 | 815.82 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 884.20 | 825.60 | 825.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 890.00 | 827.44 | 826.31 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-09 10:15:00 | 776.30 | 2025-01-10 09:15:00 | 695.14 | TARGET | 81.16 |
| BUY | 2025-04-07 15:15:00 | 818.00 | 2025-04-15 09:15:00 | 863.44 | TARGET | 45.44 |
| BUY | 2025-05-27 10:15:00 | 872.00 | 2025-05-29 10:15:00 | 860.10 | EXIT_EMA400 | -11.90 |
| BUY | 2025-07-15 10:15:00 | 864.05 | 2025-07-16 11:15:00 | 881.99 | TARGET | 17.94 |
| SELL | 2026-01-08 09:15:00 | 816.55 | 2026-01-14 15:15:00 | 840.00 | EXIT_EMA400 | -23.45 |
| SELL | 2026-03-27 09:15:00 | 773.50 | 2026-04-07 09:15:00 | 832.00 | EXIT_EMA400 | -58.50 |
