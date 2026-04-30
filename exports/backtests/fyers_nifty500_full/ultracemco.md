# UltraTech Cement Ltd. (ULTRACEMCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 11597.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -1347.12
- **Avg P&L per closed trade:** -149.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 10800.00 | 11376.83 | 11377.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 10738.95 | 11359.19 | 11368.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 11238.00 | 11229.59 | 11290.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 09:15:00 | 11014.50 | 11228.20 | 11288.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-22 10:15:00 | 11187.00 | 11059.80 | 11175.72 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 12:15:00 | 11830.00 | 11250.83 | 11248.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 13:15:00 | 11890.60 | 11257.19 | 11252.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 11495.85 | 11550.52 | 11428.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-02 12:15:00 | 11766.30 | 11500.96 | 11430.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 11485.00 | 11530.81 | 11451.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-07 09:15:00 | 11623.25 | 11531.33 | 11453.02 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 11461.95 | 11534.69 | 11458.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-08 12:15:00 | 11428.70 | 11533.63 | 11458.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 10525.35 | 11388.07 | 11391.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 13:15:00 | 10504.80 | 11379.28 | 11387.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 11249.85 | 11111.81 | 11233.81 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 13:15:00 | 11537.95 | 11304.74 | 11303.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 12:15:00 | 11559.00 | 11312.89 | 11308.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 11287.50 | 11335.36 | 11319.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-17 14:15:00 | 11497.00 | 11331.28 | 11318.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 11403.05 | 11333.30 | 11319.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-18 11:15:00 | 11310.00 | 11333.09 | 11319.76 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 11127.10 | 11309.08 | 11309.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 11043.25 | 11306.44 | 11307.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.50 | 10987.07 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 11395.10 | 11099.39 | 11098.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11620.20 | 11131.71 | 11114.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 11567.00 | 11572.00 | 11385.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 09:15:00 | 11679.00 | 11574.80 | 11391.38 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 11369.00 | 11588.84 | 11423.01 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 11259.00 | 11430.13 | 11430.18 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 11453.00 | 11430.33 | 11430.24 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 11321.00 | 11429.41 | 11429.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11425.27 | 11427.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.86 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 11767.00 | 11429.13 | 11427.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11458.38 | 11442.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12206.86 | 11965.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 09:15:00 | 12291.00 | 12208.15 | 11974.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 12390.00 | 12534.92 | 12353.43 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-12 11:15:00 | 12465.00 | 12533.09 | 12354.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-12 13:15:00 | 12353.00 | 12530.18 | 12354.63 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 12163.00 | 12306.12 | 12306.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 12136.00 | 12294.56 | 12300.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-23 13:15:00 | 12190.00 | 12290.03 | 12296.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-01 15:15:00 | 12013.00 | 11843.48 | 11992.07 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.31 | 11875.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.94 | 11878.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12669.00 | 12709.64 | 12461.45 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12299.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10762.00 | 12281.28 | 12291.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11358.66 | 11710.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 11325.00 | 11394.02 | 11693.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 11740.00 | 11402.38 | 11687.83 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 09:15:00 | 11014.50 | 2024-11-22 10:15:00 | 11187.00 | EXIT_EMA400 | -172.50 |
| BUY | 2025-01-02 12:15:00 | 11766.30 | 2025-01-08 12:15:00 | 11428.70 | EXIT_EMA400 | -337.60 |
| BUY | 2025-01-07 09:15:00 | 11623.25 | 2025-01-08 12:15:00 | 11428.70 | EXIT_EMA400 | -194.55 |
| BUY | 2025-02-17 14:15:00 | 11497.00 | 2025-02-18 11:15:00 | 11310.00 | EXIT_EMA400 | -187.00 |
| BUY | 2025-05-05 09:15:00 | 11679.00 | 2025-05-09 09:15:00 | 11369.00 | EXIT_EMA400 | -310.00 |
| BUY | 2025-07-30 09:15:00 | 12291.00 | 2025-09-12 13:15:00 | 12353.00 | EXIT_EMA400 | 62.00 |
| BUY | 2025-09-12 11:15:00 | 12465.00 | 2025-09-12 13:15:00 | 12353.00 | EXIT_EMA400 | -112.00 |
| SELL | 2025-10-23 13:15:00 | 12190.00 | 2025-10-28 14:15:00 | 11870.47 | TARGET | 319.53 |
| SELL | 2026-04-13 09:15:00 | 11325.00 | 2026-04-15 09:15:00 | 11740.00 | EXIT_EMA400 | -415.00 |
