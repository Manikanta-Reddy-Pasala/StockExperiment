# Alkem Laboratories Ltd. (ALKEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 5429.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -249.75
- **Avg P&L per closed trade:** -41.62

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 5414.85 | 5896.60 | 5897.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 5395.45 | 5891.62 | 5894.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 5735.30 | 5702.19 | 5781.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 10:15:00 | 5635.95 | 5700.90 | 5777.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-31 11:15:00 | 5673.00 | 5540.81 | 5636.05 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 5100.50 | 4977.88 | 4977.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 5124.00 | 4981.53 | 4979.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 5098.50 | 5144.49 | 5078.75 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 4836.00 | 5033.94 | 5034.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 4790.50 | 5031.52 | 5033.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 4923.00 | 4920.66 | 4966.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 09:15:00 | 4884.10 | 4921.13 | 4964.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-15 14:15:00 | 4935.80 | 4874.87 | 4923.92 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 5021.20 | 4954.57 | 4954.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 5037.90 | 4955.40 | 4954.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 4932.00 | 4958.32 | 4956.39 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 4848.00 | 4954.14 | 4954.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 4835.50 | 4952.96 | 4953.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 4928.50 | 4916.56 | 4933.74 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 5391.00 | 4950.51 | 4950.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 5404.50 | 5010.56 | 4981.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 5449.00 | 5473.70 | 5374.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-29 12:15:00 | 5504.50 | 5468.79 | 5377.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 5584.00 | 5639.35 | 5556.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 10:15:00 | 5634.00 | 5639.30 | 5556.84 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 5582.50 | 5635.72 | 5572.51 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-18 12:15:00 | 5557.50 | 5633.97 | 5572.57 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 5369.50 | 5627.10 | 5628.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 5344.00 | 5560.11 | 5587.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 11:15:00 | 5450.50 | 5443.31 | 5515.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 5392.50 | 5442.71 | 5513.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 5440.00 | 5366.93 | 5450.50 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-13 09:15:00 | 5396.00 | 5367.85 | 5450.13 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 5463.00 | 5369.09 | 5447.93 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-03 10:15:00 | 5635.95 | 2024-12-31 11:15:00 | 5673.00 | EXIT_EMA400 | -37.05 |
| SELL | 2025-07-01 09:15:00 | 4884.10 | 2025-07-15 14:15:00 | 4935.80 | EXIT_EMA400 | -51.70 |
| BUY | 2025-10-29 12:15:00 | 5504.50 | 2025-12-18 12:15:00 | 5557.50 | EXIT_EMA400 | 53.00 |
| BUY | 2025-12-09 10:15:00 | 5634.00 | 2025-12-18 12:15:00 | 5557.50 | EXIT_EMA400 | -76.50 |
| SELL | 2026-03-27 09:15:00 | 5392.50 | 2026-04-15 09:15:00 | 5463.00 | EXIT_EMA400 | -70.50 |
| SELL | 2026-04-13 09:15:00 | 5396.00 | 2026-04-15 09:15:00 | 5463.00 | EXIT_EMA400 | -67.00 |
