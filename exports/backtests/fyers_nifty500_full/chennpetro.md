# Chennai Petroleum Corporation Ltd. (CHENNPETRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1124.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 5 |
| EXIT | 5 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 323.48
- **Avg P&L per closed trade:** 32.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 13:15:00 | 901.50 | 969.12 | 969.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 09:15:00 | 889.10 | 966.97 | 968.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 15:15:00 | 930.00 | 927.84 | 943.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-26 09:15:00 | 915.80 | 927.72 | 943.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 932.45 | 926.71 | 942.23 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-03 09:15:00 | 922.95 | 927.69 | 941.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-10-04 09:15:00 | 946.25 | 927.43 | 941.07 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 603.30 | 566.06 | 565.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 608.10 | 566.48 | 566.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 560.40 | 571.94 | 568.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 10:15:00 | 570.35 | 571.92 | 568.98 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 570.35 | 571.92 | 568.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 11:15:00 | 564.25 | 571.84 | 568.95 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 14:15:00 | 656.25 | 676.91 | 676.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 648.00 | 671.97 | 674.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 668.55 | 668.51 | 672.32 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 700.00 | 675.68 | 675.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 14:15:00 | 702.65 | 675.95 | 675.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 744.75 | 757.21 | 731.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-28 11:15:00 | 802.60 | 757.87 | 737.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 901.55 | 939.80 | 886.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-10 09:15:00 | 947.30 | 938.74 | 887.91 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 899.20 | 933.53 | 894.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-18 13:15:00 | 890.75 | 932.03 | 894.77 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 810.40 | 873.91 | 873.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 801.50 | 871.41 | 872.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 860.10 | 854.11 | 863.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 827.00 | 854.56 | 862.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 858.50 | 850.38 | 859.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-22 12:15:00 | 833.20 | 850.05 | 859.55 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-27 09:15:00 | 866.25 | 848.74 | 858.36 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 919.85 | 863.40 | 863.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 933.40 | 867.27 | 865.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 872.25 | 874.60 | 869.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-17 12:15:00 | 884.10 | 874.73 | 869.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 902.35 | 918.13 | 898.02 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-12 10:15:00 | 947.25 | 918.42 | 898.26 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 927.95 | 920.04 | 900.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 11:15:00 | 942.50 | 920.37 | 900.74 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-04-13 15:15:00 | 942.00 | 970.59 | 943.72 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-26 09:15:00 | 915.80 | 2024-10-04 09:15:00 | 946.25 | EXIT_EMA400 | -30.45 |
| SELL | 2024-10-03 09:15:00 | 922.95 | 2024-10-04 09:15:00 | 946.25 | EXIT_EMA400 | -23.30 |
| BUY | 2025-04-07 10:15:00 | 570.35 | 2025-04-07 11:15:00 | 564.25 | EXIT_EMA400 | -6.10 |
| BUY | 2025-10-28 11:15:00 | 802.60 | 2025-11-06 09:15:00 | 999.11 | TARGET | 196.51 |
| BUY | 2025-12-10 09:15:00 | 947.30 | 2025-12-18 13:15:00 | 890.75 | EXIT_EMA400 | -56.55 |
| SELL | 2026-01-20 09:15:00 | 827.00 | 2026-01-27 09:15:00 | 866.25 | EXIT_EMA400 | -39.25 |
| SELL | 2026-01-22 12:15:00 | 833.20 | 2026-01-27 09:15:00 | 866.25 | EXIT_EMA400 | -33.05 |
| BUY | 2026-02-17 12:15:00 | 884.10 | 2026-02-25 09:15:00 | 927.53 | TARGET | 43.43 |
| BUY | 2026-03-16 11:15:00 | 942.50 | 2026-03-17 14:15:00 | 1067.77 | TARGET | 125.27 |
| BUY | 2026-03-12 10:15:00 | 947.25 | 2026-03-23 09:15:00 | 1094.21 | TARGET | 146.96 |
