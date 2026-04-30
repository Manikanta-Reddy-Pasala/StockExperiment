# Zydus Lifesciences Ltd. (ZYDUSLIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 891.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 113.98
- **Avg P&L per closed trade:** 19.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 591.70 | 610.25 | 610.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 12:15:00 | 587.10 | 609.14 | 609.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 595.50 | 591.37 | 598.63 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 14:15:00 | 636.90 | 604.02 | 603.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 10:15:00 | 642.30 | 605.05 | 604.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 962.50 | 966.85 | 906.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-02 09:15:00 | 978.00 | 956.23 | 919.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 977.45 | 1015.46 | 977.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 1045.05 | 1133.44 | 1133.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 1041.10 | 1116.29 | 1124.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 983.30 | 981.30 | 1016.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 09:15:00 | 975.30 | 981.36 | 1015.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 986.05 | 976.52 | 994.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-03 11:15:00 | 978.80 | 976.63 | 994.44 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-07 10:15:00 | 1009.50 | 976.01 | 993.00 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 931.80 | 901.80 | 901.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 938.55 | 903.69 | 902.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 965.55 | 966.55 | 946.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-10 12:15:00 | 970.80 | 966.61 | 946.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 957.10 | 969.53 | 953.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-22 11:15:00 | 961.85 | 969.45 | 953.24 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 951.40 | 972.60 | 958.73 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 930.00 | 991.75 | 991.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 928.20 | 973.17 | 981.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 925.00 | 924.66 | 940.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 13:15:00 | 915.20 | 925.10 | 939.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-09 13:15:00 | 917.95 | 899.02 | 914.16 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 930.50 | 906.40 | 906.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 957.00 | 908.03 | 907.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 913.15 | 914.04 | 910.44 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-02 09:15:00 | 978.00 | 2024-05-21 09:15:00 | 1153.19 | TARGET | 175.19 |
| SELL | 2024-12-04 09:15:00 | 975.30 | 2025-01-07 10:15:00 | 1009.50 | EXIT_EMA400 | -34.20 |
| SELL | 2025-01-03 11:15:00 | 978.80 | 2025-01-07 10:15:00 | 1009.50 | EXIT_EMA400 | -30.70 |
| BUY | 2025-07-22 11:15:00 | 961.85 | 2025-07-28 09:15:00 | 987.68 | TARGET | 25.83 |
| BUY | 2025-07-10 12:15:00 | 970.80 | 2025-08-01 09:15:00 | 951.40 | EXIT_EMA400 | -19.40 |
| SELL | 2026-01-08 13:15:00 | 915.20 | 2026-02-09 13:15:00 | 917.95 | EXIT_EMA400 | -2.75 |
