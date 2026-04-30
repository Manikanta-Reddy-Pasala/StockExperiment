# JSW Steel Ltd. (JSWSTEEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1264.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -15.36
- **Avg P&L per closed trade:** -2.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 755.20 | 784.80 | 784.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 753.55 | 784.49 | 784.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 785.40 | 782.22 | 783.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-13 09:15:00 | 774.60 | 782.28 | 783.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-10-16 09:15:00 | 785.10 | 781.99 | 783.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 815.40 | 772.61 | 772.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 11:15:00 | 821.65 | 773.55 | 772.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 14:15:00 | 837.20 | 838.08 | 815.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-05 09:15:00 | 843.80 | 838.14 | 816.80 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-10 09:15:00 | 815.70 | 835.95 | 817.79 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 14:15:00 | 785.90 | 816.29 | 816.42 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 870.15 | 815.92 | 815.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 880.15 | 821.19 | 818.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 10:15:00 | 839.90 | 843.41 | 832.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-19 12:15:00 | 850.95 | 843.49 | 832.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 853.15 | 863.54 | 847.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-07 13:15:00 | 860.50 | 863.42 | 847.97 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 850.50 | 863.04 | 848.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-09 12:15:00 | 843.10 | 862.73 | 848.59 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 921.85 | 967.88 | 968.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 919.75 | 966.54 | 967.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 924.80 | 923.60 | 939.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 11:15:00 | 915.35 | 923.73 | 938.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 932.45 | 923.99 | 937.73 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-24 10:15:00 | 946.15 | 924.21 | 937.77 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 11:15:00 | 961.85 | 942.99 | 942.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 13:15:00 | 966.85 | 943.44 | 943.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 955.00 | 955.30 | 950.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-03 12:15:00 | 976.35 | 955.28 | 950.27 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 930.65 | 1019.38 | 995.34 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1076.90 | 1134.90 | 1135.07 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1189.00 | 1132.07 | 1131.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1213.50 | 1159.18 | 1148.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1225.40 | 1231.46 | 1203.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 15:15:00 | 1248.10 | 1230.77 | 1205.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 1186.60 | 1230.95 | 1206.20 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 11:15:00 | 1119.50 | 1190.49 | 1190.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1117.30 | 1181.70 | 1186.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1186.40 | 1168.62 | 1178.35 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 1241.70 | 1185.98 | 1185.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 1250.00 | 1186.62 | 1186.16 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-13 09:15:00 | 774.60 | 2023-10-16 09:15:00 | 785.10 | EXIT_EMA400 | -10.50 |
| BUY | 2024-01-05 09:15:00 | 843.80 | 2024-01-10 09:15:00 | 815.70 | EXIT_EMA400 | -28.10 |
| BUY | 2024-04-19 12:15:00 | 850.95 | 2024-04-25 14:15:00 | 905.65 | TARGET | 54.70 |
| BUY | 2024-05-07 13:15:00 | 860.50 | 2024-05-09 12:15:00 | 843.10 | EXIT_EMA400 | -17.40 |
| SELL | 2025-01-22 11:15:00 | 915.35 | 2025-01-24 10:15:00 | 946.15 | EXIT_EMA400 | -30.80 |
| BUY | 2025-03-03 12:15:00 | 976.35 | 2025-03-21 09:15:00 | 1054.60 | TARGET | 78.25 |
| BUY | 2026-03-05 15:15:00 | 1248.10 | 2026-03-09 09:15:00 | 1186.60 | EXIT_EMA400 | -61.50 |
