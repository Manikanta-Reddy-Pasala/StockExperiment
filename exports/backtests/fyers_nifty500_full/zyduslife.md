# Zydus Lifesciences Ltd. (ZYDUSLIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 893.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -89.35
- **Avg P&L per closed trade:** -14.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1081.55 | 1137.76 | 1137.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1058.85 | 1136.97 | 1137.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 983.30 | 981.10 | 1016.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 10:15:00 | 969.05 | 981.04 | 1014.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 986.05 | 976.51 | 994.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-03 11:15:00 | 978.80 | 976.62 | 994.47 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-07 10:15:00 | 1009.60 | 975.96 | 993.01 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 931.80 | 901.81 | 901.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 938.50 | 903.68 | 902.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 965.55 | 966.52 | 946.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-10 12:15:00 | 970.80 | 966.59 | 946.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 957.10 | 969.53 | 953.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-22 11:15:00 | 961.85 | 969.46 | 953.23 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 951.40 | 972.52 | 958.69 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 930.00 | 991.70 | 991.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 928.20 | 973.13 | 981.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 925.00 | 924.65 | 940.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 13:15:00 | 915.20 | 925.10 | 939.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 901.10 | 897.38 | 915.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 12:15:00 | 896.25 | 897.50 | 915.14 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-09 13:15:00 | 917.95 | 898.21 | 913.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 931.50 | 906.06 | 906.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 957.45 | 909.17 | 907.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 913.15 | 914.88 | 910.78 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-04 10:15:00 | 969.05 | 2025-01-07 10:15:00 | 1009.60 | EXIT_EMA400 | -40.55 |
| SELL | 2025-01-03 11:15:00 | 978.80 | 2025-01-07 10:15:00 | 1009.60 | EXIT_EMA400 | -30.80 |
| BUY | 2025-07-22 11:15:00 | 961.85 | 2025-07-28 09:15:00 | 987.70 | TARGET | 25.85 |
| BUY | 2025-07-10 12:15:00 | 970.80 | 2025-08-01 09:15:00 | 951.40 | EXIT_EMA400 | -19.40 |
| SELL | 2026-01-08 13:15:00 | 915.20 | 2026-02-09 13:15:00 | 917.95 | EXIT_EMA400 | -2.75 |
| SELL | 2026-02-03 12:15:00 | 896.25 | 2026-02-09 13:15:00 | 917.95 | EXIT_EMA400 | -21.70 |
