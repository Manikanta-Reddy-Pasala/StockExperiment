# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 14699.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 23.41
- **Avg P&L per closed trade:** 2.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 14193.35 | 15815.72 | 15816.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 14:15:00 | 14149.40 | 15285.33 | 15506.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 09:15:00 | 14722.80 | 14712.98 | 15082.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 09:15:00 | 13993.15 | 14817.91 | 15006.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-27 11:15:00 | 14300.00 | 13701.66 | 14110.08 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 17300.00 | 14472.27 | 14463.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 14:15:00 | 17669.60 | 15410.62 | 14986.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 13:15:00 | 15916.60 | 15960.90 | 15373.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-24 15:15:00 | 16150.00 | 15933.63 | 15404.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-08 12:15:00 | 15451.00 | 15876.95 | 15520.42 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 15387.85 | 15569.51 | 15570.15 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 15827.55 | 15571.10 | 15570.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 09:15:00 | 16245.50 | 15585.54 | 15577.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 15950.00 | 16136.11 | 15920.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-27 15:15:00 | 16533.60 | 16125.08 | 15935.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 15950.00 | 16123.31 | 15937.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-30 12:15:00 | 15929.35 | 16121.38 | 15937.35 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 15064.50 | 15809.41 | 15810.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 14934.00 | 15699.22 | 15752.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 14938.50 | 14744.09 | 15131.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-22 14:15:00 | 14411.50 | 14775.81 | 15039.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 14935.00 | 14773.18 | 15029.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-26 11:15:00 | 14599.25 | 14772.32 | 15023.99 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-07 12:15:00 | 11888.20 | 11140.09 | 11855.03 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 12654.95 | 11625.41 | 11620.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 13467.25 | 11643.73 | 11629.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 13:15:00 | 12109.00 | 12163.52 | 11931.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 09:15:00 | 12455.10 | 12164.53 | 11935.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-02 11:15:00 | 12183.00 | 12550.73 | 12259.24 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 12561.00 | 13488.78 | 13493.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 12540.00 | 13462.21 | 13479.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 10:15:00 | 13501.00 | 13342.43 | 13414.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-17 13:15:00 | 12896.00 | 13276.03 | 13358.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 13064.00 | 12886.74 | 13066.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-27 10:15:00 | 13050.00 | 12890.04 | 13034.14 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 14937.00 | 13155.00 | 13152.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 15258.00 | 13880.46 | 13579.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 14471.00 | 14576.66 | 14145.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-30 11:15:00 | 15093.00 | 14213.00 | 14083.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 14935.00 | 15206.53 | 14812.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 11:15:00 | 14735.00 | 15198.11 | 14811.90 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 13600.00 | 14561.40 | 14563.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 13545.00 | 14497.72 | 14530.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 14465.00 | 14119.03 | 14309.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-06 09:15:00 | 13800.00 | 14127.03 | 14301.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 14542.00 | 14078.85 | 14264.40 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 15144.00 | 14370.92 | 14370.37 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-15 09:15:00 | 13993.15 | 2024-05-27 11:15:00 | 14300.00 | EXIT_EMA400 | -306.85 |
| BUY | 2024-06-24 15:15:00 | 16150.00 | 2024-07-08 12:15:00 | 15451.00 | EXIT_EMA400 | -699.00 |
| BUY | 2024-09-27 15:15:00 | 16533.60 | 2024-09-30 12:15:00 | 15929.35 | EXIT_EMA400 | -604.25 |
| SELL | 2024-11-22 14:15:00 | 14411.50 | 2024-11-27 09:15:00 | 12526.13 | TARGET | 1885.37 |
| SELL | 2024-11-26 11:15:00 | 14599.25 | 2024-11-27 09:15:00 | 13325.02 | TARGET | 1274.23 |
| BUY | 2025-04-11 09:15:00 | 12455.10 | 2025-05-02 11:15:00 | 12183.00 | EXIT_EMA400 | -272.10 |
| SELL | 2025-10-17 13:15:00 | 12896.00 | 2025-11-27 10:15:00 | 13050.00 | EXIT_EMA400 | -154.00 |
| BUY | 2026-01-30 11:15:00 | 15093.00 | 2026-03-02 11:15:00 | 14735.00 | EXIT_EMA400 | -358.00 |
| SELL | 2026-04-06 09:15:00 | 13800.00 | 2026-04-08 09:15:00 | 14542.00 | EXIT_EMA400 | -742.00 |
