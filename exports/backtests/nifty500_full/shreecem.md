# Shree Cement Ltd. (SHREECEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 24195.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / EMA400 exits:** 2 / 10
- **Total realized P&L (per unit):** -2536.72
- **Avg P&L per closed trade:** -211.39

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 14:15:00 | 25966.90 | 24462.59 | 24460.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 15:15:00 | 26035.10 | 24661.53 | 24563.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 11:15:00 | 25405.50 | 25473.90 | 25088.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-03 12:15:00 | 25659.95 | 25476.17 | 25104.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-25 12:15:00 | 25488.30 | 25958.77 | 25552.85 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 13:15:00 | 26527.65 | 27263.88 | 27265.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 09:15:00 | 26250.00 | 27194.69 | 27230.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 26000.00 | 25697.53 | 26223.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-28 10:15:00 | 25726.75 | 25715.27 | 26212.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-01 11:15:00 | 26424.50 | 25734.23 | 26202.16 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 09:15:00 | 27236.75 | 25673.08 | 25672.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 10:15:00 | 27476.70 | 25691.03 | 25681.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 27008.70 | 27022.45 | 26541.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-09 09:15:00 | 27467.85 | 27033.75 | 26560.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 27111.05 | 27422.28 | 26990.42 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-30 11:15:00 | 27163.40 | 27419.70 | 26991.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 26531.40 | 27440.08 | 27054.57 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 24300.00 | 26747.99 | 26750.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 24226.00 | 26631.01 | 26691.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 25684.75 | 25544.41 | 25995.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-02 12:15:00 | 25458.15 | 25544.78 | 25988.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-05 09:15:00 | 26089.10 | 25552.49 | 25954.21 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 27371.80 | 25275.85 | 25271.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 27450.00 | 25316.94 | 25291.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 09:15:00 | 26582.80 | 26603.86 | 26097.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 09:15:00 | 26861.50 | 26400.19 | 26080.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 26140.00 | 26392.57 | 26084.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-03 15:15:00 | 26184.40 | 26390.49 | 26085.28 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 26120.00 | 26385.80 | 26085.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-06 11:15:00 | 26070.25 | 26382.66 | 26085.88 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 25238.05 | 25904.01 | 25904.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 25098.05 | 25871.68 | 25888.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 14:15:00 | 25905.95 | 25806.05 | 25852.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 25337.40 | 25810.02 | 25852.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-28 12:15:00 | 26080.85 | 25776.48 | 25833.52 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 26790.75 | 25888.46 | 25887.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 27000.00 | 25916.34 | 25901.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 27273.20 | 27608.29 | 27013.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-05 14:15:00 | 28183.60 | 27594.31 | 27077.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 29140.00 | 29946.44 | 29125.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-02 12:15:00 | 29085.00 | 29937.87 | 29125.56 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 29915.00 | 30443.95 | 30444.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 29710.00 | 30353.37 | 30397.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 30350.00 | 30335.83 | 30386.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-10 09:15:00 | 30075.00 | 30338.86 | 30385.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 30075.00 | 30338.86 | 30385.39 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-10 10:15:00 | 29955.00 | 30335.04 | 30383.24 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 29900.00 | 29900.98 | 30103.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-01 09:15:00 | 29085.00 | 29852.29 | 30069.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-15 12:15:00 | 29920.00 | 29656.47 | 29890.08 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-03 12:15:00 | 25659.95 | 2023-10-25 12:15:00 | 25488.30 | EXIT_EMA400 | -171.65 |
| SELL | 2024-03-28 10:15:00 | 25726.75 | 2024-04-01 11:15:00 | 26424.50 | EXIT_EMA400 | -697.75 |
| BUY | 2024-07-30 11:15:00 | 27163.40 | 2024-07-31 11:15:00 | 27679.76 | TARGET | 516.36 |
| BUY | 2024-07-09 09:15:00 | 27467.85 | 2024-08-05 09:15:00 | 26531.40 | EXIT_EMA400 | -936.45 |
| SELL | 2024-09-02 12:15:00 | 25458.15 | 2024-09-05 09:15:00 | 26089.10 | EXIT_EMA400 | -630.95 |
| BUY | 2025-01-03 09:15:00 | 26861.50 | 2025-01-06 11:15:00 | 26070.25 | EXIT_EMA400 | -791.25 |
| BUY | 2025-01-03 15:15:00 | 26184.40 | 2025-01-06 11:15:00 | 26070.25 | EXIT_EMA400 | -114.15 |
| SELL | 2025-01-27 09:15:00 | 25337.40 | 2025-01-28 12:15:00 | 26080.85 | EXIT_EMA400 | -743.45 |
| BUY | 2025-03-05 14:15:00 | 28183.60 | 2025-05-02 12:15:00 | 29085.00 | EXIT_EMA400 | 901.40 |
| SELL | 2025-09-10 09:15:00 | 30075.00 | 2025-09-26 09:15:00 | 29143.83 | TARGET | 931.17 |
| SELL | 2025-09-10 10:15:00 | 29955.00 | 2025-10-15 12:15:00 | 29920.00 | EXIT_EMA400 | 35.00 |
| SELL | 2025-10-01 09:15:00 | 29085.00 | 2025-10-15 12:15:00 | 29920.00 | EXIT_EMA400 | -835.00 |
