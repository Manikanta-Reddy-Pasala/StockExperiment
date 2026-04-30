# UltraTech Cement Ltd. (ULTRACEMCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 11586.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** 256.53
- **Avg P&L per closed trade:** 25.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 9373.40 | 9743.61 | 9744.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 9307.00 | 9735.71 | 9740.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 9692.30 | 9686.18 | 9712.14 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 9993.70 | 9734.84 | 9734.29 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 14:15:00 | 9514.15 | 9734.36 | 9734.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 09:15:00 | 9468.30 | 9729.52 | 9732.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 13:15:00 | 9685.15 | 9684.02 | 9707.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-15 09:15:00 | 9611.75 | 9682.93 | 9706.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-16 14:15:00 | 9714.15 | 9674.29 | 9700.94 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 10165.75 | 9725.28 | 9724.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 10226.25 | 9734.65 | 9729.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 9842.95 | 9900.58 | 9824.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-07 11:15:00 | 10322.35 | 9926.46 | 9845.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-14 09:15:00 | 11038.35 | 11436.33 | 11152.15 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 10776.45 | 11382.30 | 11384.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 10698.05 | 11118.98 | 11221.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 11151.90 | 11059.52 | 11179.21 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 11890.60 | 11257.79 | 11254.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 11977.85 | 11264.96 | 11258.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 11495.85 | 11551.06 | 11430.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-02 11:15:00 | 11672.75 | 11498.50 | 11430.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 11485.00 | 11530.91 | 11453.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-07 09:15:00 | 11623.25 | 11531.56 | 11454.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 11461.95 | 11534.92 | 11459.42 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-08 12:15:00 | 11428.70 | 11533.86 | 11459.27 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 10561.45 | 11396.93 | 11397.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 10525.35 | 11388.26 | 11392.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 11249.85 | 11111.64 | 11234.45 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 11559.00 | 11304.75 | 11304.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 11590.00 | 11313.09 | 11308.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 11287.50 | 11328.66 | 11316.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-17 14:15:00 | 11497.00 | 11325.46 | 11315.68 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 11403.05 | 11327.65 | 11316.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-18 11:15:00 | 11310.00 | 11327.55 | 11316.94 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 11145.40 | 11307.19 | 11307.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 11043.25 | 11302.79 | 11305.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.16 | 10986.36 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 11395.10 | 11098.50 | 11097.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11620.15 | 11131.19 | 11114.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 11567.00 | 11572.00 | 11385.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 09:15:00 | 11679.00 | 11574.67 | 11391.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 11369.00 | 11589.82 | 11423.28 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 11266.00 | 11429.40 | 11429.88 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 11493.00 | 11430.46 | 11430.39 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 11321.00 | 11430.14 | 11430.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11425.99 | 11428.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 11435.00 | 11410.94 | 11420.34 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 11765.00 | 11430.17 | 11428.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11459.16 | 11443.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12207.48 | 11966.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 09:15:00 | 12291.00 | 12208.63 | 11975.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 12391.00 | 12535.50 | 12354.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-12 11:15:00 | 12466.00 | 12533.67 | 12354.88 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-12 13:15:00 | 12353.00 | 12530.75 | 12355.20 | Close below EMA400 |

### Cycle 15 — SELL (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 14:15:00 | 12177.00 | 12307.18 | 12307.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 12036.00 | 12290.03 | 12298.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 12289.00 | 12278.55 | 12292.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-23 13:15:00 | 12189.00 | 12290.76 | 12296.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-26 10:15:00 | 11815.00 | 11658.21 | 11802.05 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.28 | 11875.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.91 | 11877.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12662.00 | 12704.29 | 12453.58 | EMA200 retest candle locked |

### Cycle 17 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11089.00 | 12293.72 | 12293.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10765.00 | 12278.51 | 12286.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11357.16 | 11706.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 11325.00 | 11393.13 | 11690.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 11735.00 | 11401.50 | 11684.78 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-15 09:15:00 | 9611.75 | 2024-05-16 14:15:00 | 9714.15 | EXIT_EMA400 | -102.40 |
| BUY | 2024-06-07 11:15:00 | 10322.35 | 2024-06-27 09:15:00 | 11754.01 | TARGET | 1431.66 |
| BUY | 2025-01-02 11:15:00 | 11672.75 | 2025-01-08 12:15:00 | 11428.70 | EXIT_EMA400 | -244.05 |
| BUY | 2025-01-07 09:15:00 | 11623.25 | 2025-01-08 12:15:00 | 11428.70 | EXIT_EMA400 | -194.55 |
| BUY | 2025-02-17 14:15:00 | 11497.00 | 2025-02-18 11:15:00 | 11310.00 | EXIT_EMA400 | -187.00 |
| BUY | 2025-05-05 09:15:00 | 11679.00 | 2025-05-09 09:15:00 | 11369.00 | EXIT_EMA400 | -310.00 |
| BUY | 2025-07-30 09:15:00 | 12291.00 | 2025-09-12 13:15:00 | 12353.00 | EXIT_EMA400 | 62.00 |
| BUY | 2025-09-12 11:15:00 | 12466.00 | 2025-09-12 13:15:00 | 12353.00 | EXIT_EMA400 | -113.00 |
| SELL | 2025-10-23 13:15:00 | 12189.00 | 2025-11-04 12:15:00 | 11865.13 | TARGET | 323.87 |
| SELL | 2026-04-13 09:15:00 | 11325.00 | 2026-04-15 09:15:00 | 11735.00 | EXIT_EMA400 | -410.00 |
