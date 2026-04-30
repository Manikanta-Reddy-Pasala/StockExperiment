# The Ramco Cements Ltd. (RAMCOCEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 929.00
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
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 60.19
- **Avg P&L per closed trade:** 10.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 863.15 | 933.00 | 933.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 852.00 | 902.50 | 914.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 10:15:00 | 874.00 | 869.33 | 889.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 11:15:00 | 865.45 | 870.22 | 888.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 865.10 | 857.29 | 875.52 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 15:15:00 | 864.95 | 857.97 | 875.33 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-27 12:15:00 | 879.30 | 858.46 | 874.64 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 948.00 | 886.40 | 886.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 955.15 | 891.54 | 888.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 935.20 | 937.19 | 919.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 949.50 | 937.35 | 920.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-08 09:15:00 | 1070.60 | 1135.87 | 1090.62 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 1045.00 | 1075.08 | 1075.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1024.10 | 1063.97 | 1068.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1030.40 | 1027.25 | 1044.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-20 09:15:00 | 1014.25 | 1026.95 | 1042.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-21 13:15:00 | 1043.25 | 1026.88 | 1042.33 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 1061.40 | 1030.42 | 1030.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1074.70 | 1033.96 | 1032.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1057.50 | 1059.04 | 1048.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-20 11:15:00 | 1070.70 | 1059.56 | 1049.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1068.00 | 1059.86 | 1049.40 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-21 09:15:00 | 1074.60 | 1060.00 | 1049.53 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-23 15:15:00 | 1046.00 | 1062.54 | 1051.87 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 978.80 | 1084.18 | 1084.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 946.70 | 1078.76 | 1081.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 987.00 | 985.57 | 1022.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 10:15:00 | 965.00 | 991.69 | 1015.38 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-10 11:15:00 | 865.45 | 2025-03-12 10:15:00 | 795.84 | TARGET | 69.61 |
| SELL | 2025-03-25 15:15:00 | 864.95 | 2025-03-27 12:15:00 | 879.30 | EXIT_EMA400 | -14.35 |
| BUY | 2025-05-12 09:15:00 | 949.50 | 2025-06-09 13:15:00 | 1036.73 | TARGET | 87.23 |
| SELL | 2025-10-20 09:15:00 | 1014.25 | 2025-10-21 13:15:00 | 1043.25 | EXIT_EMA400 | -29.00 |
| BUY | 2026-01-20 11:15:00 | 1070.70 | 2026-01-23 15:15:00 | 1046.00 | EXIT_EMA400 | -24.70 |
| BUY | 2026-01-21 09:15:00 | 1074.60 | 2026-01-23 15:15:00 | 1046.00 | EXIT_EMA400 | -28.60 |
