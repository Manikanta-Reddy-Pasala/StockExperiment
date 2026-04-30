# Tata Elxsi Ltd. (TATAELXSI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4132.40
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
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -25.38
- **Avg P&L per closed trade:** -3.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 13:15:00 | 8744.00 | 7024.54 | 7021.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 14:15:00 | 9045.00 | 7044.64 | 7031.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 12:15:00 | 7517.50 | 7553.78 | 7374.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-20 09:15:00 | 7643.00 | 7553.76 | 7377.64 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 7575.00 | 7665.09 | 7490.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-04 10:15:00 | 7610.15 | 7664.54 | 7491.58 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 7584.50 | 7660.44 | 7492.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-07 09:15:00 | 7449.30 | 7657.35 | 7493.05 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 7081.55 | 7435.04 | 7435.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 7006.75 | 7427.07 | 7431.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 7101.00 | 6882.21 | 7068.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-01 10:15:00 | 6734.70 | 7029.78 | 7091.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 5520.50 | 5260.31 | 5576.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 11:15:00 | 5577.50 | 5266.32 | 5576.37 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 6234.00 | 5720.65 | 5718.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 6283.00 | 5830.94 | 5777.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6295.00 | 6295.97 | 6111.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 12:15:00 | 6333.00 | 6293.10 | 6122.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 6162.00 | 6290.33 | 6152.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-02 09:15:00 | 6148.00 | 6285.54 | 6153.75 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 5828.00 | 6134.85 | 6135.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 5802.00 | 6131.54 | 6133.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 5733.50 | 5650.67 | 5815.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 09:15:00 | 5546.50 | 5674.86 | 5785.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 5591.50 | 5465.16 | 5582.66 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 10:15:00 | 5661.00 | 5346.01 | 5345.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 11:15:00 | 5758.50 | 5350.11 | 5347.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 12:15:00 | 5409.00 | 5426.07 | 5390.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-22 09:15:00 | 5464.00 | 5420.19 | 5388.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5464.00 | 5420.19 | 5388.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-23 13:15:00 | 5380.00 | 5420.92 | 5390.97 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 5097.50 | 5373.91 | 5375.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 5025.00 | 5370.44 | 5373.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4421.50 | 4378.91 | 4644.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 4378.10 | 4381.71 | 4637.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 4531.70 | 4401.54 | 4614.26 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 11:15:00 | 4524.00 | 4404.20 | 4613.49 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 4591.90 | 4408.94 | 4612.76 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 09:15:00 | 4665.00 | 4413.37 | 4612.95 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-20 09:15:00 | 7643.00 | 2024-10-07 09:15:00 | 7449.30 | EXIT_EMA400 | -193.70 |
| BUY | 2024-10-04 10:15:00 | 7610.15 | 2024-10-07 09:15:00 | 7449.30 | EXIT_EMA400 | -160.85 |
| SELL | 2025-01-01 10:15:00 | 6734.70 | 2025-02-27 09:15:00 | 5663.63 | TARGET | 1071.07 |
| BUY | 2025-06-23 12:15:00 | 6333.00 | 2025-07-02 09:15:00 | 6148.00 | EXIT_EMA400 | -185.00 |
| SELL | 2025-09-23 09:15:00 | 5546.50 | 2025-10-27 09:15:00 | 5591.50 | EXIT_EMA400 | -45.00 |
| BUY | 2026-01-22 09:15:00 | 5464.00 | 2026-01-23 13:15:00 | 5380.00 | EXIT_EMA400 | -84.00 |
| SELL | 2026-04-09 09:15:00 | 4378.10 | 2026-04-17 09:15:00 | 4665.00 | EXIT_EMA400 | -286.90 |
| SELL | 2026-04-16 11:15:00 | 4524.00 | 2026-04-17 09:15:00 | 4665.00 | EXIT_EMA400 | -141.00 |
