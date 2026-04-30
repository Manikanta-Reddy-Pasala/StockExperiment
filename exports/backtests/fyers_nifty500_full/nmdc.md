# NMDC Ltd. (NMDC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 90.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -5.58
- **Avg P&L per closed trade:** -1.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 15:15:00 | 78.15 | 75.92 | 75.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 78.29 | 75.95 | 75.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 77.03 | 77.34 | 76.70 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 70.86 | 76.24 | 76.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 70.05 | 75.18 | 75.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 68.20 | 68.08 | 70.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 66.68 | 68.08 | 70.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 67.15 | 64.96 | 67.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-21 09:15:00 | 66.60 | 65.02 | 67.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 66.94 | 65.05 | 67.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-21 12:15:00 | 67.57 | 65.08 | 67.31 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 70.02 | 66.51 | 66.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 70.50 | 66.73 | 66.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 70.68 | 70.72 | 69.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 71.45 | 69.86 | 69.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 09:15:00 | 67.73 | 69.89 | 69.22 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 78.10 | 79.91 | 79.91 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 80.76 | 79.92 | 79.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 80.97 | 79.93 | 79.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 79.74 | 79.94 | 79.93 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 75.45 | 79.89 | 79.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 75.19 | 79.84 | 79.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 79.96 | 78.87 | 79.32 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 14:15:00 | 84.49 | 79.74 | 79.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 84.71 | 79.84 | 79.77 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-24 09:15:00 | 66.68 | 2025-02-21 12:15:00 | 67.57 | EXIT_EMA400 | -0.89 |
| SELL | 2025-02-21 09:15:00 | 66.60 | 2025-02-21 12:15:00 | 67.57 | EXIT_EMA400 | -0.97 |
| BUY | 2025-06-27 09:15:00 | 71.45 | 2025-07-01 09:15:00 | 67.73 | EXIT_EMA400 | -3.72 |
